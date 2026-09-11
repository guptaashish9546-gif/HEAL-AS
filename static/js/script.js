document.addEventListener("DOMContentLoaded", () => {
  const menu = document.querySelector(".navmenu");
  const links = document.querySelector(".navlinks");
  if (menu && links) menu.addEventListener("click", () => links.classList.toggle("open"));

  const forms = ["diabetes-form", "dengue-form"];
  forms.forEach(id => {
    const form = document.getElementById(id);
    if (!form) return;
    form.addEventListener("submit", async e => {
      e.preventDefault();
      const result = document.getElementById("result");
      const button = form.querySelector("button[type='submit']");
      const original = button.innerHTML;
      button.disabled = true;
      button.innerHTML = `<span class="spinner"></span> Analysing...`;
      result.classList.remove("hidden");
      result.innerHTML = `<p><strong>Processing securely...</strong> The model is running and your PDF report is being prepared.</p>`;
      try {
        const response = await fetch(form.action, { method: "POST", body: new FormData(form) });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || "Analysis failed.");
        const confidence = data.confidence == null ? null : Number(data.confidence);
        const badgeClass = data.prediction === "YES" ? "yes" : "no";
        const guidance = data.prediction === "YES"
          ? "Please consult a qualified healthcare professional for proper evaluation and diagnosis."
          : "A negative AI prediction does not guarantee the absence of disease.";
        result.innerHTML = `
          <div class="result-title">
            <div><span class="eyebrow">ANALYSIS COMPLETE</span><h3>${data.disease} screening result</h3></div>
            <span class="result-badge ${badgeClass}">${data.prediction}</span>
          </div>
          ${confidence != null ? `<p><strong>Model confidence:</strong> ${confidence.toFixed(2)}%</p><div class="progress"><span style="width:${Math.min(100, Math.max(0, confidence))}%"></span></div>` : ""}
          <p>${guidance}</p>
          <a class="btn small" href="${data.download_url}">Download PDF Report</a>`;
      } catch (err) {
        result.innerHTML = `<div class="error"><strong>Unable to complete analysis.</strong><br>${err.message}</div>`;
      } finally {
        button.disabled = false;
        button.innerHTML = original;
      }
    });
  });

  const xray = document.getElementById("xray");
  const preview = document.getElementById("preview");
  const fileName = document.getElementById("file-name");
  if (xray && preview) {
    xray.addEventListener("change", () => {
      const file = xray.files[0];
      if (!file) return;
      if (!file.type.startsWith("image/")) { xray.value = ""; return; }
      preview.src = URL.createObjectURL(file);
      preview.classList.remove("hidden");
      if (fileName) fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    });
  }
});
