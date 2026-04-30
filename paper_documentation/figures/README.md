# Figures (manuscript & web)

Vector and raster assets for the paper and static site. Colours match the results UI (`#0f1419` background, `#5b9fd4` accent).

## `cgse_system_workflow.svg` — Figure 4.5 (system workflow)

**Type:** Non-linear **evolution loop** diagram (not a single left-to-right chain). Setup is a vertical stack; the core is a **closed clockwise cycle** (Training → Evaluation → Mutation decision → Apply mutation → Continue training) with a **green feedback arc** back to Training (`repeat until stop`), plus a **dashed exit** to **Final model** when a stopping criterion is met.

**Embed in Markdown / HTML (centered):**

```html
<p align="center">
  <img src="figures/cgse_system_workflow.svg" alt="CGSE system workflow" width="820" />
</p>
<p align="center"><em>Figure 4.5: System workflow.</em></p>
```

**LaTeX (pdfLaTeX with `svg` not native — export PDF from Inkscape or use `pdf_tex`, or include PNG export):**

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/cgse_system_workflow.pdf}
  \caption{System workflow: setup feeds an evolution loop (training through mutation application); the loop repeats until a stopping criterion is satisfied, then yields the final model.}
  \label{fig:workflow}
\end{figure}
```

For repositories that only accept raster in CI, convert once:  
`inkscape figures/cgse_system_workflow.svg --export-type=png --export-filename=cgse_system_workflow.png`

The website build script copies `*.png` and `*.svg` from this folder into `web/assets/figures/`.
