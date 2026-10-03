// Chat: POST /api/ask and render the server-sent event stream (token / sources / error / done).

const form = document.getElementById("ask");
const filters = document.getElementById("filters");
const messages = document.getElementById("messages");
const textarea = form.querySelector("textarea");
const button = form.querySelector("button");

function addMessage(cls, text = "") {
  messages.querySelector(".empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text;
  messages.appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
  return el;
}

function renderSources(el, sources) {
  if (!sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  sources.forEach((s, i) => {
    const d = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `[${i + 1}] ${s.subject} · Class ${s.class_num} · Page ${s.page_num} — ${s.source}`;
    const quote = document.createElement("blockquote");
    quote.textContent = s.text;
    d.append(summary, quote);
    box.appendChild(d);
  });
  el.appendChild(box);
}

function buildBody(question) {
  const f = new FormData(filters);
  return {
    question,
    class_num: f.get("class_num") ? Number(f.get("class_num")) : null,
    subject: f.get("subject") || null,
    language: f.get("language") || null,
  };
}

async function ask(question) {
  addMessage("user", question);
  const bot = addMessage("bot pending");
  const answer = document.createTextNode("");
  bot.appendChild(answer);

  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildBody(question)),
  });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const events = buffer.split("\n\n");
    buffer = events.pop();
    for (const raw of events) {
      const event = raw.match(/^event: (.*)$/m)?.[1];
      const data = JSON.parse(raw.match(/^data: (.*)$/m)?.[1] ?? "null");
      if (event === "token") answer.appendData(data);
      else if (event === "sources") renderSources(bot, data);
      else if (event === "error") { bot.classList.add("error"); answer.appendData(data); }
    }
    bot.scrollIntoView({ block: "end" });
  }
  bot.classList.remove("pending");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = textarea.value.trim();
  if (!question) return;
  textarea.value = "";
  button.disabled = true;
  try {
    await ask(question);
  } catch (err) {
    addMessage("bot error", err.message);
  } finally {
    messages.querySelectorAll(".pending").forEach((el) => el.classList.remove("pending"));
    button.disabled = false;
    textarea.focus();
  }
});

// Enter sends, Shift+Enter adds a new line.
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
