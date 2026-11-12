#!/usr/bin/env python3
"""Téléchargement et préparation des données du projet (culture : TOMATE).

Ce script est la seule source de vérité pour les données : le notebook ne lit
que les manifestes CSV produits ici.

Sorties (dans --data-dir, par défaut ./data) :
  plantvillage_tomato/<classe>/*.jpg   images PlantVillage (tomate) extraites
  raw/PlantDoc-Dataset/...             PlantDoc (tomate uniquement, clone partiel)
  manifest_plantvillage.csv            path,label,group,split   (split : train/val/test)
  manifest_plantdoc.csv                path,label,orig_label,split
  data_report.json                     comptages, doublons retirés, contrôles de fuite

Choix méthodologiques (à reprendre dans le rapport) :
  * Espace de classes commun aux deux jeux : 8 classes (voir CLASSES).
    - « Target spot » est retirée : absente de PlantDoc.
    - « Spider mites » est retirée : 2 images seulement dans PlantDoc (train), 0 en test.
  * PlantVillage : jeu Hugging Face `geraldmc/plantvillage-full` (réédition des images de
    `raw/color/` du dépôt PlantVillage, avec identifiant de feuille et split train/test groupé
    par feuille). La validation est prélevée dans le train, par feuille (StratifiedGroupKFold) :
    aucune feuille n'est partagée entre train, val et test. Attention : le regroupement n'existe
    que pour les classes où `leaf_grouped` vaut True ; le script liste les classes concernées.
  * PlantDoc : le split officiel train/test est conservé. Les images strictement identiques
    (hash MD5) sont dédupliquées ; si les copies ont des étiquettes contradictoires, toutes
    les copies sont retirées. La validation est prélevée dans le train après dédoublonnage.

Usage :
  python scripts/download_data.py [--data-dir data] [--seed 42]
                                  [--skip-plantvillage] [--skip-plantdoc]
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset"
PLANTVILLAGE_HF = "geraldmc/plantvillage-full"   # 54 304 images, split train/test groupé par feuille (CC0)
PLANTVILLAGE_REV = "v0.1.0"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = [
    "healthy",
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "yellow_leaf_curl_virus",
    "mosaic_virus",
]


def canonical(name):
    """Nom de dossier PlantVillage / PlantDoc -> classe commune (ou None si exclue)."""
    s = name.lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if "tomato" not in s:
        return None
    if "target" in s or "mite" in s:  # classes exclues (voir en-tête)
        return None
    if "healthy" in s or s == "tomato leaf":
        return "healthy"
    if "bacterial" in s:
        return "bacterial_spot"
    if "early" in s:
        return "early_blight"
    if "late" in s and "blight" in s:
        return "late_blight"
    if "mold" in s:
        return "leaf_mold"
    if "septoria" in s:
        return "septoria_leaf_spot"
    if "yellow" in s:
        return "yellow_leaf_curl_virus"
    if "mosaic" in s:
        return "mosaic_virus"
    return None


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- PlantDoc
def prepare_plantdoc(data_dir, seed, report):
    from sklearn.model_selection import train_test_split

    raw = data_dir / "raw" / "PlantDoc-Dataset"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print("[PlantDoc] clone partiel (dossiers Tomato uniquement)...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                            PLANTDOC_URL, str(raw)], check=True)
            subprocess.run(["git", "-C", str(raw), "sparse-checkout", "set", "--no-cone",
                            "/train/Tomato*", "/test/Tomato*"], check=True)
        except subprocess.CalledProcessError:
            print("[PlantDoc] clone partiel impossible, clone complet (~2 Go)...")
            subprocess.run(["rm", "-rf", str(raw)])
            subprocess.run(["git", "clone", "--depth", "1", PLANTDOC_URL, str(raw)], check=True)

    rows = []
    for split in ("train", "test"):
        for cls_dir in sorted((raw / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            lab = canonical(cls_dir.name)
            if lab is None:
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append(dict(path=str(f.relative_to(data_dir)), label=lab,
                                     orig_label=cls_dir.name, split=split, md5=md5(f)))
    df = pd.DataFrame(rows)
    report["plantdoc_before_dedup"] = df.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_dict()

    # --- doublons exacts (contrôle de fuite) ---
    removed = []
    keep = pd.Series(True, index=df.index)
    for _, g in df.groupby("md5"):
        if len(g) == 1:
            continue
        if g["label"].nunique() > 1:  # étiquettes contradictoires : on retire tout
            keep[g.index] = False
            kind = "etiquettes_contradictoires"
        else:  # même étiquette : on garde une copie, de préférence celle du test
            g_sorted = g.sort_values("split", ascending=False)  # 'test' avant 'train'
            keep[g_sorted.index[1:]] = False
            kind = "copie_identique"
        removed.append(dict(type=kind, fichiers=g["path"].tolist(), splits=g["split"].tolist(),
                            labels=g["label"].tolist()))
    df = df[keep].reset_index(drop=True)
    report["plantdoc_duplicates_removed"] = removed
    print(f"[PlantDoc] {len(removed)} groupes de doublons traités")

    # --- validation prélevée dans le train (après dédoublonnage) ---
    tr = df[df.split == "train"]
    tr_idx, va_idx = train_test_split(tr.index, test_size=0.15, random_state=seed,
                                      stratify=tr["label"])
    df.loc[va_idx, "split"] = "val"
    df = df.drop(columns=["md5"])
    assert not (set(df[df.split == "test"].path) & set(df[df.split != "test"].path))
    df.to_csv(data_dir / "manifest_plantdoc.csv", index=False)
    report["plantdoc_final"] = df.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_dict()
    print(df.groupby(["label", "split"]).size().unstack(1).fillna(0).astype(int))


# ------------------------------------------------------------------------ PlantVillage
def prepare_plantvillage(data_dir, seed, report):
    from datasets import load_dataset
    from sklearn.model_selection import StratifiedGroupKFold

    print("[PlantVillage] chargement depuis Hugging Face (~850 Mo, plusieurs minutes)...")
    try:
        ds = load_dataset(PLANTVILLAGE_HF, revision=PLANTVILLAGE_REV, split="train")
    except Exception as e:  # révision introuvable : on essaie la branche principale
        print(f"  révision {PLANTVILLAGE_REV} indisponible ({type(e).__name__}), essai sur la branche principale")
        ds = load_dataset(PLANTVILLAGE_HF, split="train")
    need = ["image", "class_label", "split", "leaf_id"]
    missing = [c for c in need if c not in ds.column_names]
    if missing:
        raise RuntimeError(f"Colonnes attendues absentes : {missing}. Colonnes disponibles : {ds.column_names}")

    # lecture des colonnes de métadonnées SEULES (sans décoder les images)
    labels, splits, leaves = list(ds["class_label"]), list(ds["split"]), list(ds["leaf_id"])
    grouped = list(ds["leaf_grouped"]) if "leaf_grouped" in ds.column_names else [None] * len(ds)
    idxs, canon = [], []
    for i, s_ in enumerate(labels):
        c = canonical(s_)
        if c is not None:
            idxs.append(i); canon.append(c)
    print(f"[PlantVillage] {len(idxs)} images de tomate retenues sur {len(ds)}")

    out_root = data_dir / "plantvillage_tomato"
    rows = []
    for j, ex in enumerate(ds.select(idxs)):
        i, c = idxs[j], canon[j]
        out = out_root / c / f"{splits[i]}_{i:06d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            ex["image"].convert("RGB").save(out, quality=95)
        rows.append(dict(path=str(out.relative_to(data_dir)), label=c, group=leaves[i],
                         leaf_grouped=grouped[i], split=splits[i]))
    df = pd.DataFrame(rows)
    assert set(df.split) <= {"train", "test"}, f"valeurs de split inattendues : {set(df.split)}"

    # --- le regroupement par feuille est-il réel pour nos classes ? ---
    if df.leaf_grouped.notna().all():
        share = df.groupby("label")["leaf_grouped"].mean().round(3).to_dict()
        report["plantvillage_leaf_grouped_share_by_class"] = share
        ungrouped = sorted(k for k, v in share.items() if v < 1)
        print("[PlantVillage] part d'images avec regroupement par feuille réel :", share)
        if ungrouped:
            print("  ATTENTION : classes SANS regroupement par feuille (un identifiant par image) :", ungrouped,
                  "\n  -> risque de fuite (photos d'une même feuille dans train et test) pour ces classes ;"
                  "\n     à signaler dans la section « Limites et risques » du rapport.")

    # --- validation prélevée dans le train, PAR FEUILLE ---
    tr = df[df.split == "train"]
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    _, va_pos = next(sgkf.split(tr, tr["label"], groups=tr["group"]))
    df.loc[tr.index[va_pos], "split"] = "val"

    groups = {s_: set(df[df.split == s_].group) for s_ in ("train", "val", "test")}
    overlap = {f"{a}/{b}": len(groups[a] & groups[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    report["plantvillage_leaf_overlap"] = overlap
    assert all(v == 0 for v in overlap.values()), f"Fuite de feuilles détectée : {overlap}"
    df.to_csv(data_dir / "manifest_plantvillage.csv", index=False)
    report["plantvillage_final"] = df.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_dict()
    print(df.groupby(["label", "split"]).size().unstack(1).fillna(0).astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-plantvillage", action="store_true")
    ap.add_argument("--skip-plantdoc", action="store_true")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    report = {"classes": CLASSES, "seed": a.seed}
    if not a.skip_plantdoc:
        prepare_plantdoc(data_dir, a.seed, report)
    if not a.skip_plantvillage:
        prepare_plantvillage(data_dir, a.seed, report)
    (data_dir / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("Terminé. Rapport : ", data_dir / "data_report.json")


if __name__ == "__main__":
    main()
