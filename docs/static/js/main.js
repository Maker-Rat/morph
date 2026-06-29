const copyButtons = document.querySelectorAll("[data-copy-target]");

copyButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    const text = target.innerText;
    try {
      await navigator.clipboard.writeText(text);
      const previous = button.innerText;
      button.innerText = "Copied";
      setTimeout(() => {
        button.innerText = previous;
      }, 1400);
    } catch (error) {
      button.innerText = "Select text";
    }
  });
});
