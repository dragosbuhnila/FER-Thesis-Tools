import os

folder_latek = "Figures/Comparisons"
# saliency_maps\zzz_other_and_zips\output_comparisons_meandif\comparisons_occlusion
folder_local = "saliency_maps/zzz_other_and_zips/output_comparisons_meandif/comparisons_occlusion"

ungrouped = []
grouped = []

for f in sorted(os.listdir(folder_local)):
    if not f.endswith(".png"):
        continue
    
    if "methods_vertical" in f:
        grouped.append(f)
    else:
        ungrouped.append(f)

def clean_caption(filename):
    name = filename.replace("Comparison_", "")
    name = name.replace("_merged", "")
    name = name.replace(".png", "")
    
    parts = name.split("_VS_")
    left = parts[0].replace("_", " ")
    right = parts[1].replace("_", " ")
    
    return left, right


latex = []

latex.append("\\subsection{Complete Humans vs Machines Comparisons Ungrouped}\n")

for f in ungrouped:
    left, right = clean_caption(f)

    caption = f"Human vs machine comparison: {left} vs {right}."

    latex.append(f"""
\\begin{{figure}}[p]
\\centering
\\includegraphics[width=\\textwidth]{{Figures/Comparisons/{f}}}
\\caption{{{caption}}}
\\end{{figure}}
""")

latex.append("\n\\subsection{Complete Humans vs Machines Comparisons Grouped By Model}\n")

for f in grouped:
    left, right = clean_caption(f.replace("_methods_vertical", ""))

    caption = f"Human vs machine comparison: {left} vs {right} including Bubbles, External Perturbations and GradCAM."

    latex.append(f"""
\\begin{{sidewaysfigure}}[htbp]
\\centering
\\includegraphics[width=\\textheight]{{Figures/Comparisons/{f}}}
\\caption{{{caption}}}
\\end{{sidewaysfigure}}
""")

with open("appendix_figures.tex", "w", encoding="utf8") as out:
    out.write("\n".join(latex))

print("LaTeX code written to appendix_figures.tex")