const copyButtons = document.querySelectorAll("[data-copy-target]");
const pageVideos = document.querySelectorAll("video");
const themeToggle = document.querySelector(".theme-toggle");
const themeToggleLabel = document.querySelector(".theme-toggle-label");

function activeTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function updateThemeToggle() {
  if (!themeToggle) return;
  const isDark = activeTheme() === "dark";
  themeToggle.setAttribute("aria-pressed", String(isDark));
  themeToggle.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
  if (themeToggleLabel) {
    themeToggleLabel.textContent = isDark ? "Light" : "Dark";
  }
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("xmorph-theme", theme);
  } catch (error) {
    // Theme persistence is optional; the toggle should still work.
  }
  updateThemeToggle();
}

function prepareVideo(video) {
  video.muted = true;
  video.defaultMuted = true;
  video.loop = true;
  video.autoplay = true;
  video.playsInline = true;
  video.controls = false;
  video.setAttribute("muted", "");
  video.setAttribute("playsinline", "");
  video.setAttribute("webkit-playsinline", "");
  video.setAttribute("disablepictureinpicture", "");
  video.setAttribute("controlslist", "nodownload noplaybackrate noremoteplayback");
}

function playVideo(video) {
  prepareVideo(video);
  const playPromise = video.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {
      video.dataset.autoplayBlocked = "true";
    });
  }
}

function playAllVideos() {
  pageVideos.forEach(playVideo);
}

pageVideos.forEach(playVideo);
updateThemeToggle();

["pointerdown", "touchstart", "click"].forEach((eventName) => {
  document.addEventListener(eventName, playAllVideos, { once: true, passive: true });
});

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    setTheme(activeTheme() === "dark" ? "light" : "dark");
  });
}

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
