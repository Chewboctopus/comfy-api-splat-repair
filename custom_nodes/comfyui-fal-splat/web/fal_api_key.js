/**
 * comfyui-fal-splat — FAL API Key Manager
 *
 * On ComfyUI startup:
 *   1. Checks /fal-splat/api-key/status
 *   2. If no key is configured, shows a modal dialog asking for it
 *   3. Saves the key via /fal-splat/api-key/save (writes to config.ini + env)
 *   4. Key persists across restarts via config.ini
 *
 * Also adds a "FAL API Key" option to the ComfyUI settings menu
 * so users can update their key later.
 */

import { app } from "../../scripts/app.js";

const EXTENSION_NAME = "fal-splat.apiKeyManager";

// ── Styles ──────────────────────────────────────────────────────────────────

const MODAL_STYLES = `
  .fal-key-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.65);
    z-index: 99999;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }

  .fal-key-modal {
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 12px;
    padding: 28px 32px;
    width: 480px;
    max-width: 90vw;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    color: #cdd6f4;
  }

  .fal-key-modal h2 {
    margin: 0 0 8px 0;
    font-size: 20px;
    color: #f5c2e7;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .fal-key-modal p {
    margin: 0 0 16px 0;
    font-size: 14px;
    color: #a6adc8;
    line-height: 1.5;
  }

  .fal-key-modal a {
    color: #89b4fa;
    text-decoration: none;
  }

  .fal-key-modal a:hover {
    text-decoration: underline;
  }

  .fal-key-input-row {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
  }

  .fal-key-input {
    flex: 1;
    padding: 10px 14px;
    background: #313244;
    border: 1px solid #585b70;
    border-radius: 8px;
    color: #cdd6f4;
    font-size: 14px;
    font-family: "SF Mono", "Fira Code", monospace;
    outline: none;
    transition: border-color 0.2s;
  }

  .fal-key-input:focus {
    border-color: #89b4fa;
  }

  .fal-key-input::placeholder {
    color: #6c7086;
  }

  .fal-key-btn {
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .fal-key-btn-primary {
    background: #89b4fa;
    color: #1e1e2e;
  }

  .fal-key-btn-primary:hover {
    background: #b4d0fb;
  }

  .fal-key-btn-primary:disabled {
    background: #585b70;
    color: #6c7086;
    cursor: not-allowed;
  }

  .fal-key-btn-secondary {
    background: #45475a;
    color: #cdd6f4;
  }

  .fal-key-btn-secondary:hover {
    background: #585b70;
  }

  .fal-key-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 4px;
  }

  .fal-key-status {
    font-size: 13px;
    min-height: 20px;
  }

  .fal-key-status.success { color: #a6e3a1; }
  .fal-key-status.error   { color: #f38ba8; }
  .fal-key-status.loading { color: #f9e2af; }

  .fal-key-current {
    background: #313244;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 16px;
    font-size: 13px;
    color: #a6adc8;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .fal-key-current .key-value {
    font-family: "SF Mono", "Fira Code", monospace;
    color: #a6e3a1;
  }
`;

// ── Modal Logic ─────────────────────────────────────────────────────────────

async function checkKeyStatus() {
  try {
    const resp = await fetch("/fal-splat/api-key/status");
    return await resp.json();
  } catch (e) {
    console.warn("[fal-splat] Could not check API key status:", e);
    return { has_key: false, masked_key: "" };
  }
}

async function saveKey(key) {
  const resp = await fetch("/fal-splat/api-key/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  return await resp.json();
}

function createModal(existingKey = "") {
  // Inject styles
  if (!document.getElementById("fal-key-styles")) {
    const style = document.createElement("style");
    style.id = "fal-key-styles";
    style.textContent = MODAL_STYLES;
    document.head.appendChild(style);
  }

  const overlay = document.createElement("div");
  overlay.className = "fal-key-overlay";

  const isUpdate = !!existingKey;
  const title = isUpdate ? "Update FAL API Key" : "FAL API Key Required";
  const description = isUpdate
    ? "Your current key is shown below. Enter a new key to replace it."
    : "The <b>fal-splat</b> nodes need a fal.ai API key to call Flux, Qwen, and other models. Your key is saved locally in <code>config.ini</code> and persists across restarts.";

  let currentKeyHtml = "";
  if (existingKey) {
    currentKeyHtml = `
      <div class="fal-key-current">
        <span>Current key:</span>
        <span class="key-value">${existingKey}</span>
      </div>
    `;
  }

  overlay.innerHTML = `
    <div class="fal-key-modal">
      <h2>🔑 ${title}</h2>
      <p>${description}</p>
      ${currentKeyHtml}
      <div class="fal-key-input-row">
        <input
          type="password"
          class="fal-key-input"
          id="fal-key-input"
          placeholder="Paste your fal.ai API key here..."
          spellcheck="false"
          autocomplete="off"
        />
        <button class="fal-key-btn fal-key-btn-primary" id="fal-key-save">
          Save
        </button>
      </div>
      <div class="fal-key-footer">
        <span class="fal-key-status" id="fal-key-status"></span>
        <div style="display: flex; gap: 8px;">
          <a href="https://fal.ai/dashboard/keys" target="_blank" class="fal-key-btn fal-key-btn-secondary" style="text-decoration:none; display:inline-block;">
            Get Key ↗
          </a>
          ${isUpdate ? '<button class="fal-key-btn fal-key-btn-secondary" id="fal-key-cancel">Cancel</button>' : '<button class="fal-key-btn fal-key-btn-secondary" id="fal-key-skip">Skip for Now</button>'}
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  const input = overlay.querySelector("#fal-key-input");
  const saveBtn = overlay.querySelector("#fal-key-save");
  const statusEl = overlay.querySelector("#fal-key-status");
  const skipBtn = overlay.querySelector("#fal-key-skip") || overlay.querySelector("#fal-key-cancel");

  // Toggle password visibility on double-click
  input.addEventListener("dblclick", () => {
    input.type = input.type === "password" ? "text" : "password";
  });

  // Focus input
  setTimeout(() => input.focus(), 100);

  // Save handler
  saveBtn.addEventListener("click", async () => {
    const key = input.value.trim();
    if (!key) {
      statusEl.className = "fal-key-status error";
      statusEl.textContent = "Please enter a key.";
      return;
    }

    saveBtn.disabled = true;
    statusEl.className = "fal-key-status loading";
    statusEl.textContent = "Validating and saving...";

    try {
      const result = await saveKey(key);
      if (result.ok) {
        statusEl.className = "fal-key-status success";
        statusEl.textContent = "✓ " + result.message + " Saved to config.ini.";
        setTimeout(() => overlay.remove(), 1500);
      } else {
        statusEl.className = "fal-key-status error";
        statusEl.textContent = "✗ " + result.message;
        saveBtn.disabled = false;
      }
    } catch (e) {
      statusEl.className = "fal-key-status error";
      statusEl.textContent = "Network error: " + e.message;
      saveBtn.disabled = false;
    }
  });

  // Enter key to save
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") saveBtn.click();
  });

  // Skip/cancel
  if (skipBtn) {
    skipBtn.addEventListener("click", () => overlay.remove());
  }

  return overlay;
}

// ── ComfyUI Extension Registration ──────────────────────────────────────────

app.registerExtension({
  name: EXTENSION_NAME,

  async setup() {
    // Check on startup
    const status = await checkKeyStatus();

    if (!status.has_key) {
      // No key configured — show the modal
      console.log("[fal-splat] No FAL API key found, showing setup dialog.");
      createModal();
    } else {
      console.log("[fal-splat] FAL API key configured:", status.masked_key);
    }

    // Add a menu item so users can update the key later.
    // We hook into the ComfyUI menu by adding a button to the settings area.
    try {
      // Try the ComfyUI settings API (varies by version)
      const settingsBtn = document.createElement("button");
      settingsBtn.textContent = "🔑 FAL API Key";
      settingsBtn.title = "Configure your fal.ai API key";
      settingsBtn.style.cssText = `
        background: #313244; color: #cdd6f4; border: 1px solid #45475a;
        border-radius: 6px; padding: 4px 10px; font-size: 12px; cursor: pointer;
        margin: 2px;
      `;
      settingsBtn.addEventListener("click", async () => {
        const s = await checkKeyStatus();
        createModal(s.masked_key || "");
      });

      // Try to find the ComfyUI menu/toolbar area
      const tryAppend = () => {
        const menuRight = document.querySelector(".comfyui-menu-right")
          || document.querySelector(".comfy-menu-btns")
          || document.querySelector(".comfyui-body-top .flex.gap-2");
        if (menuRight) {
          menuRight.appendChild(settingsBtn);
          return true;
        }
        return false;
      };

      // Retry a few times since the menu may not be ready yet
      if (!tryAppend()) {
        let attempts = 0;
        const interval = setInterval(() => {
          if (tryAppend() || ++attempts > 20) clearInterval(interval);
        }, 500);
      }
    } catch (e) {
      console.warn("[fal-splat] Could not add settings button:", e);
    }
  },
});
