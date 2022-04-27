katex_options = {
macros: {"\\i":               "\\mathrm{i}",
"\\e":             "\\mathrm{e}^{#1}",
"\\vec":           "\\mathbf{#1}",
"\\x":               "\\vec{x}",
"\\d":               "\\operatorname{d}\\!",
"\\dirac":         "\\operatorname{\\delta}\\left(#1\\right)",
"\\scalarprod":  "\\left\\langle#1,#2\\right\\rangle",
"\\timeStep":         "\\delta_t",
"\\nrSnapshots":      "M",
"\\nrSnapshotsGen":   "\\tilde{\\nrSnapshots}",
"\\state":            "x",
"\\stateVar":         "\\state",
"\\stateDim":         "n",
"\\stateDimRed":      "r",
"\\inpVar":           "u",
"\\inputVar":         "\\inpVar",
"\\inpVarDim":        "m",
"\\inputDim":         "\\inpVarDim",
"\\outVar":           "y",
"\\outputVar":        "\\outVar",
"\\dmdV":             "X_+",
"\\dmdW":             "X_{-}",
"\\dmdY":             "Y",
"\\dmdU":             "U",
"\\dataZ":            "\\mathcal{Z}",
"\\dataT":            "\\mathcal{T}",
"\\dmdJJ":            "\\widetilde{\\mathcal{J}}",
"\\dmdRR":            "\\widetilde{\\mathcal{R}}",},
delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
        ]
}
document.addEventListener("DOMContentLoaded", function() {
  renderMathInElement(document.body, katex_options);
});
