import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.16.0/dist/mermaid.esm.min.mjs";

const codeBlocks = Array.from(
  document.querySelectorAll(".post-content pre > code.language-mermaid"),
);

if (codeBlocks.length > 0) {
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "neutral",
    fontFamily:
      'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif',
    flowchart: {
      htmlLabels: false,
      useMaxWidth: true,
    },
    sequence: {
      useMaxWidth: true,
    },
  });

  const diagrams = [];

  for (const [index, code] of codeBlocks.entries()) {
    const source = code.textContent.trim();

    try {
      await mermaid.parse(source);

      const container = document.createElement("div");
      container.className = "mermaid mermaid-diagram";
      container.id = `mermaid-diagram-${index + 1}`;
      container.setAttribute("role", "img");
      container.setAttribute("aria-label", `Mermaid diagram ${index + 1}`);
      container.textContent = source;

      code.parentElement.replaceWith(container);
      diagrams.push(container);
    } catch (error) {
      code.parentElement.classList.add("mermaid-source-error");
      console.error("Unable to parse Mermaid diagram", error);
    }
  }

  if (diagrams.length > 0) {
    try {
      await mermaid.run({ nodes: diagrams });
    } catch (error) {
      console.error("Unable to render Mermaid diagrams", error);
    }
  }
}
