const $ = (id) => document.getElementById(id);
let selected = null;
let current = null;
let tab = "chapters";
let librarySignature = "";
let selectionVersion = 0;
const labels = {
  chapters: "Главы · TXT",
  chapters_json: "Главы · JSON",
  transcript: "Расшифровка · TXT",
  result_json: "Whisper · JSON",
};
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = "Не удалось выполнить запрос.";
    try {
      message = (await response.json()).error || message;
    } catch {}
    throw new Error(message);
  }
  return response.json();
}
function date(value) {
  return new Date(value).toLocaleDateString("ru-RU", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}
function renderJobs(jobs) {
  const visible = jobs.filter((j) => j.status !== "done");
  const failures = visible.filter((j) => j.status === "failed");
  const expanded = document.querySelector(".failed-jobs")?.open || false;
  $("queue-section").hidden = !visible.length;
  $("queue-title").textContent = visible.some((j) => j.status !== "failed")
    ? "В работе"
    : "История обработки";
  $("jobs").replaceChildren();
  const failedGroup = el("details", "failed-jobs");
  failedGroup.open = expanded;
  failedGroup.append(
    el("summary", "", `Неудачные попытки · ${failures.length}`),
  );
  for (const job of visible) {
    const row = el("div", `job ${job.status}`), info = el("div", "job-info");
    info.append(
      el("strong", "", job.title),
      el("p", "", job.error || job.stage),
    );
    const log = el("button", "text-button", "Журнал ↗");
    log.onclick = async () => {
      try {
        const data = await api(`/api/jobs/${job.id}/log`);
        $("log-text").textContent = data.text;
        $("log-dialog").showModal();
      } catch (error) {
        showConnection(error);
      }
    };
    row.append(el("span", "job-dot"), info, log);
    (job.status === "failed" ? failedGroup : $("jobs")).append(row);
  }
  if (failures.length) $("jobs").append(failedGroup);
}
function renderLibrary(results) {
  $("result-count").textContent = results.length;
  $("results").replaceChildren();
  for (const result of results) {
    const button = el(
      "button",
      `result ${selected === result.id ? "active" : ""}`,
    );
    button.dataset.id = result.id;
    button.setAttribute("aria-pressed", String(selected === result.id));
    const meta = el("div", "result-meta");
    meta.append(
      el("span", "", date(result.created)),
      el(
        "span",
        "badge",
        result.files.includes("chapters")
          ? "Главы готовы"
          : "Расшифровка готова",
      ),
    );
    if (result.preview) meta.append(el("span", "", "Первые 5 минут"));
    button.append(
      el("div", "result-title", result.title.replaceAll("_", " ")),
      meta,
    );
    button.onclick = () => selectResult(result.id);
    $("results").append(button);
  }
}
async function selectResult(id) {
  const version = ++selectionVersion;
  try {
    const result = await api(`/api/results/${id}`);
    if (version !== selectionVersion) return;
    selected = id;
    current = result;
    localStorage.setItem("selected-result", id);
    for (const node of $("results").children) {
      node.classList.toggle("active", node.dataset.id === id);
      node.setAttribute("aria-pressed", String(node.dataset.id === id));
    }
    renderDetail();
  } catch (error) {
    showConnection(error);
  }
}
function renderDetail() {
  const container = $("detail");
  container.replaceChildren();
  const top = el("div", "detail-top");
  top.append(
    el(
      "small",
      "",
      `${date(current.created)}${current.preview ? " · Первые 5 минут" : ""}`,
    ),
    el("h2", "", current.title.replaceAll("_", " ")),
  );
  const tabs = el("div", "tabs");
  for (
    const [key, name] of [["chapters", "Таймкоды"], [
      "transcript",
      "Расшифровка",
    ]]
  ) {
    const button = el("button", `tab ${tab === key ? "active" : ""}`, name);
    button.onclick = () => {
      tab = key;
      renderDetail();
    };
    tabs.append(button);
  }
  const copy = el("button", "text-button copy", "Копировать");
  copy.disabled = !current[tab];
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(current[tab]);
      copy.textContent = "Скопировано ✓";
    } catch {
      copy.textContent = "Не удалось скопировать";
    }
  };
  tabs.append(copy);
  container.append(top, tabs);
  if (!current[tab]) {
    container.append(el(
      "p",
      "muted",
      tab === "chapters"
        ? "Главы пока не готовы."
        : "Расшифровка пока недоступна.",
    ));
  } else if (tab === "transcript") {
    container.append(el("pre", "transcript", current.transcript));
  } else {
    for (const line of current.chapters.split("\n").filter((l) => l.trim())) {
      const match = line.match(/^(\d{2}:\d{2}:\d{2})\s+(.*)$/);
      const row = el("div", "chapter");
      row.append(
        el("time", "", match ? match[1] : ""),
        el("span", "", match ? match[2] : line),
      );
      container.append(row);
    }
  }
  const downloads = el("div", "downloads");
  for (const kind of current.files) {
    const a = el("a", "", `↓ ${labels[kind]}`);
    a.href = `/api/results/${current.id}/download/${kind}`;
    downloads.append(a);
  }
  container.append(downloads);
}
function showConnection(error) {
  $("connection-error").hidden = false;
  $("connection-error").textContent = error.message +
    " Проверьте, что локальный сервер запущен.";
}
async function refresh() {
  try {
    const [jobs, results] = await Promise.all([
      api("/api/jobs"),
      api("/api/results"),
    ]);
    $("connection-error").hidden = true;
    renderJobs(jobs);
    const signature = JSON.stringify(results);
    if (signature !== librarySignature) {
      librarySignature = signature;
      renderLibrary(results);
      const remembered = selected || localStorage.getItem("selected-result");
      const id = results.some((r) => r.id === remembered)
        ? remembered
        : results[0]?.id;
      if (id) await selectResult(id);
    }
  } catch (error) {
    showConnection(error);
  }
}
$("new-job").onsubmit = async (event) => {
  event.preventDefault();
  $("submit").disabled = true;
  const message = $("form-message");
  message.hidden = true;
  message.className = "";
  try {
    await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: $("url").value,
        disk_path: $("disk-path").value,
      }),
    });
    message.textContent = "Видео добавлено в очередь.";
    message.hidden = false;
    await refresh();
  } catch (error) {
    message.textContent = error.message;
    message.hidden = false;
    message.className = "error";
  } finally {
    $("submit").disabled = false;
  }
};
$("refresh").onclick = async () => {
  librarySignature = "";
  await refresh();
};
$("close-log").onclick = () => $("log-dialog").close();
async function poll() {
  await refresh();
  setTimeout(poll, 3000);
}
poll();
