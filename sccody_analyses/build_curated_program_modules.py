#!/usr/bin/env python3
"""Build the gene-program modules used by the residual program analysis.

Every module is either an MSigDB gene set (name and release recorded) or a gene
list defined in a cited publication and derived by a stated rule. Modules are
stored as column indices into the AD or COVID gene panel.

Inputs
  modules/gmt/*.gmt                         MSigDB H, C2:CP:Reactome, C5:GO
  <downloads>/msig/c2.cgp, c8.all v2026.1   MSigDB C2:CGP, C8
  <downloads>/mathys2023_decontx/Exc_*.csv  Mathys et al. 2023 (Cell) DecontX/RUVr
                                            muscat results, global AD pathology (gpath),
                                            14 excitatory-neuron subtypes
Outputs (configs/programs/curated/)
  AD_modules.json, COVID_modules.json       {module: [panel column indices]}
  module_sources.tsv                        source, citation, URL, rule, coverage
  module_genes.tsv                          long table of module x gene (panel genes)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

MSIGDB = "https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/{}"
MATHYS_URL = ("https://github.com/mathyslab7/ROSMAP_snRNAseq_PFC/tree/main/Results/"
              "DecontX_RUVr_Differential_gene_expression_analysis/muscat_DecontX/Results/gpath")
MATHYS_CITE = ("Mathys H et al. Single-cell atlas reveals correlates of high cognitive function, dementia, "
               "and resilience to Alzheimer's disease pathology. Cell 186, 4365-4385 (2023)")
REYES_CITE = "Reyes M et al. An immune-cell signature of bacterial sepsis. Nat Med 26, 333-340 (2020)"
HAO_CITE = ("Hao Y et al. Integrated analysis of multimodal single-cell data. Cell 184, 3573-3587 (2021); "
            "Azimuth human PBMC reference, celltype.l2 'CD16 Mono' marker genes")
SILVIN_CITE = ("Silvin A et al. Elevated calprotectin and abnormal myeloid cell subsets discriminate "
               "severe from mild COVID-19. Cell 182, 1401-1418 (2020)")


def read_gmt(paths) -> dict[str, tuple[list[str], str]]:
    sets = {}
    for path in paths:
        release = Path(path).name.split(".v")[-1].replace(".Hs.symbols.gmt", "")
        for line in Path(path).read_text().splitlines():
            parts = line.split("\t")
            sets[parts[0]] = (parts[2:], f"MSigDB {release} ({Path(path).name.split('.v')[0]})")
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules-dir", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--mathys-min-subtypes", type=int, default=7,
                        help="same-direction per-subtype BH FDR<0.05 in >= this many of 14 EN subtypes")
    args = parser.parse_args()
    out = args.modules_dir / "curated"
    out.mkdir(parents=True, exist_ok=True)

    gmt = read_gmt(list((args.modules_dir / "gmt").glob("*.gmt")) +
                   [args.downloads / "msig/c2.cgp.v2026.1.Hs.symbols.gmt",
                    args.downloads / "msig/c8.all.v2026.1.Hs.symbols.gmt"])

    def msig(label: str, name: str) -> dict:
        genes, release = gmt[name]
        return {"module": label, "genes": genes, "source": f"{name} ({release})",
                "citation": "MSigDB; original source as listed on the gene-set page", "url": MSIGDB.format(name),
                "rule": "full gene set"}

    # Mathys 2023: majority-rule AD-pathology genes across EN subtypes.
    tables = [pd.read_csv(f) for f in glob.glob(str(args.downloads / "mathys2023_decontx/Exc_*.csv"))
              if not f.endswith("Exc_IT_gpath.csv")]
    mathys = pd.concat(tables)
    n_subtypes = mathys["cluster_id"].nunique()
    sig = mathys[mathys["p_adj.glb"] < 0.05]
    up = sig[sig["logFC"] > 0].groupby("gene").size()
    dn = sig[sig["logFC"] < 0].groupby("gene").size()
    k = args.mathys_min_subtypes
    mathys_up = sorted(set(up[up >= k].index) - set(dn.index))
    mathys_dn = sorted(set(dn[dn >= k].index) - set(up.index))
    mathys_rule = (f"per-subtype BH FDR < 0.05 (p_adj.glb, equal to p_adj.loc in these per-subtype tables) for global "
                   f"AD pathology (gpath), same direction in >= {k} of {n_subtypes} excitatory-neuron subtypes, "
                   f"no opposite-direction hits; DecontX/RUVr results")

    ad_modules = [
        msig("Synaptic signaling", "GOBP_SYNAPTIC_SIGNALING"),
        msig("Calcium ion transmembrane transport", "GOBP_CALCIUM_ION_TRANSMEMBRANE_TRANSPORT"),
        msig("Response to oxidative stress", "GOBP_RESPONSE_TO_OXIDATIVE_STRESS"),
        msig("Response to heat", "GOBP_RESPONSE_TO_HEAT"),
        msig("TNFα signaling via NF-κB", "HALLMARK_TNFA_SIGNALING_VIA_NFKB"),
        {"module": "AD-pathology EN genes, up (Mathys 2023)", "genes": mathys_up, "source": "Mathys 2023 DecontX gpath",
         "citation": MATHYS_CITE, "url": MATHYS_URL, "rule": mathys_rule},
        {"module": "AD-pathology EN genes, down (Mathys 2023)", "genes": mathys_dn, "source": "Mathys 2023 DecontX gpath",
         "citation": MATHYS_CITE, "url": MATHYS_URL, "rule": mathys_rule},
    ]
    covid_modules = [
        msig("Interferon α/β signaling", "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"),
        msig("Interferon-α response", "HALLMARK_INTERFERON_ALPHA_RESPONSE"),
        msig("Interferon-γ response", "HALLMARK_INTERFERON_GAMMA_RESPONSE"),
        msig("MHC class II protein complex", "GOCC_MHC_CLASS_II_PROTEIN_COMPLEX"),
        msig("TNFα signaling via NF-κB", "HALLMARK_TNFA_SIGNALING_VIA_NFKB"),
        msig("Inflammatory response", "HALLMARK_INFLAMMATORY_RESPONSE"),
        msig("IL-6/JAK/STAT3 signaling", "HALLMARK_IL6_JAK_STAT3_SIGNALING"),
        msig("Non-classical monocyte (Travaglini 2020)", "TRAVAGLINI_LUNG_NONCLASSICAL_MONOCYTE_CELL"),
        {"module": "MS1 monocyte score (Reyes 2020)",
         "genes": ["PLAC8", "CLU", "RETN", "CD63", "ALOX5AP", "SEC61G", "TXN", "MT1X"],
         "source": "Reyes 2020, Fig. 3d,e", "citation": REYES_CITE, "url": "https://doi.org/10.1038/s41591-020-0752-4",
         "rule": "PLAC8, CLU and the 'top 6 MS1 marker genes' as defined in the paper's Fig. 3 score"},
        {"module": "Calprotectin S100A8/A9 + S100A12", "genes": ["S100A8", "S100A9", "S100A12"],
         "source": "named genes", "citation": SILVIN_CITE, "url": "https://doi.org/10.1016/j.cell.2020.08.002",
         "rule": "calprotectin subunits and S100A12 named explicitly; min-gene threshold waived (3 genes)"},
        {"module": "Calprotectin complex (GO:1990660)", "genes": ["S100A8", "S100A9"],
         "source": "GO:1990660 calprotectin complex (QuickGO, human annotations)", "citation": "Gene Ontology Consortium",
         "url": "https://www.ebi.ac.uk/QuickGO/term/GO:1990660",
         "rule": "all human gene products annotated part_of GO:1990660 (S100A8, S100A9; retrieved 2026-10-13); "
                 "below MSigDB's size floor, so taken from GO directly; min-gene threshold waived (2 genes)"},
        {"module": "CD16 monocyte (Azimuth PBMC, Hao 2021)",
         "genes": ["CDKN1C", "FCGR3A", "PTPRC", "LST1", "IER5", "MS4A7", "RHOC", "IFITM3", "AIF1", "HES4"],
         "source": "Azimuth human PBMC reference", "citation": HAO_CITE,
         "url": "https://azimuth.hubmapconsortium.org/references/",
         "rule": "full published marker list for celltype.l2 'CD16 Mono' (blood-derived; retrieved 2026-10-13)"},
    ]

    panels = {
        "AD": pd.read_csv(args.modules_dir / "AD_panel.tsv", sep="\t"),
        "COVID": pd.read_csv(args.modules_dir / "COVID_panel.tsv", sep="\t"),
    }
    source_rows, gene_rows = [], []
    for dataset, modules in (("AD", ad_modules), ("COVID", covid_modules)):
        panel = panels[dataset]
        usable = panel["observed"].to_numpy(bool)
        index = {s: i for i, s in enumerate(panel["symbol"].astype(str)) if usable[i]}
        resolved = {}
        for spec in modules:
            hits = sorted({index[g] for g in spec["genes"] if g in index})
            keep = len(hits) >= args.min_genes or spec["module"].startswith("Calprotectin")
            if keep:
                resolved[spec["module"]] = hits
                gene_rows += [{"dataset": dataset, "module": spec["module"], "symbol": panel["symbol"].iloc[i],
                               "panel_index": i} for i in hits]
            source_rows.append({"dataset": dataset, "module": spec["module"], "source": spec["source"],
                                "citation": spec["citation"], "url": spec["url"], "rule": spec["rule"],
                                "n_set_genes": len(set(spec["genes"])), "n_panel_genes": len(hits),
                                "included": keep})
        (out / f"{dataset}_modules.json").write_text(json.dumps(resolved, indent=1) + "\n")
    pd.DataFrame(source_rows).to_csv(out / "module_sources.tsv", sep="\t", index=False)
    pd.DataFrame(gene_rows).to_csv(out / "module_genes.tsv", sep="\t", index=False)
    print(pd.DataFrame(source_rows)[["dataset", "module", "n_set_genes", "n_panel_genes", "included"]].to_string(index=False))


if __name__ == "__main__":
    main()
