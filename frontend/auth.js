const API_BASE =
  window.location.protocol.startsWith("http") && window.location.port === "8000"
    ? window.location.origin
    : "http://127.0.0.1:8000";

function apiUrl(path) {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

async function requestJson(path, options) {
  const response = await fetch(apiUrl(path), options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

const statusText = document.getElementById("authStatus");
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");

if (localStorage.getItem("staffName") && window.location.pathname.endsWith("login.html")) {
  window.location.replace("/frontend/index.html");
}

async function submitAuth(endpoint, username, password, successMessage) {
  statusText.textContent = "Please wait...";
  try {
    const result = await requestJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    localStorage.setItem("staffName", result.staff);
    statusText.textContent = successMessage;
    window.setTimeout(() => {
      window.location.replace("/frontend/index.html");
    }, 250);
  } catch (error) {
    statusText.textContent = error.message;
  }
}

loginForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitAuth(
    "/auth/login",
    document.getElementById("loginUsername").value.trim(),
    document.getElementById("loginPassword").value.trim(),
    "Login successful. Opening dashboard..."
  );
});

registerForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitAuth(
    "/auth/register",
    document.getElementById("registerUsername").value.trim(),
    document.getElementById("registerPassword").value.trim(),
    "Account created. Opening dashboard..."
  );
});
