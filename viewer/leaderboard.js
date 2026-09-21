/**
 * The board itself.
 *
 * Every value rendered here came from a stranger's issue and survived
 * tools/verify_replay.mjs. It is still treated as text: names go in through
 * textContent, never innerHTML, so a name that looks like markup renders as
 * the characters someone typed rather than as part of this page.
 */

import { loadLang, saveLang } from "./src/i18n.js";
import { LIMITS, MIN_SUBMITTABLE_WINS } from "./src/replay.js?v=long-replay-download";
import { RANKED_OPENING_DELAY_SECONDS } from "./src/ranked.js";

/**
 * The submission rules, stated once, here. Every one of them is enforced twice
 * over — the page refuses to record a run that breaks one, and the verifier
 * refuses the record independently, because the page's copy of a rule is only
 * advisory once a submission is just text somebody can write.
 */
const RULES = {
  en: [
    [`${MIN_SUBMITTABLE_WINS} win to enter`, `Every human win counts. A run needs at least `
      + `${MIN_SUBMITTABLE_WINS} win before it can be submitted.`],
    [`Up to ${LIMITS.maxRounds} rounds / ${LIMITS.maxFrames.toLocaleString()} frames`,
      "The ranking score is the total number of wins in one continuous ranked run. "
      + "Recording stops at whichever limit comes first. Losses and double KOs are recorded "
      + "alongside it. Rerolling the maze or changing the opponent starts a new run."],
    ["No delay for the opponent", "Opponent delay must sit at 0 frames. It is the "
      + "handicap that makes the agent actuate late, and a ranked run gives it none."],
    ["Opening pause at most 0.5s", `The default ${RANKED_OPENING_DELAY_SECONDS}s or `
      + "shorter. A shorter pause only makes the run harder, so it is allowed."],
    ["No turn-rate assist", "Instant turn is switched off when a ranked run starts and "
      + "has to stay off."],
    ["Three boards", "Hybrid, Laika and Killfield are ranked separately."],
    ["Anything else is yours", "The wheel's forward region, touch or keyboard, pausing "
      + "to think — none of it changes the match, so none of it is restricted."],
    ["A name, and optionally an account", "Every record goes up under a name. Showing "
      + "a GitHub account is optional. In one-click submissions it is a self-reported "
      + "profile link, not proof that the player owns that account."],
    ["Submit or keep the replay", "Large records are split automatically when submitted. "
      + "Every finished ranked run can also be downloaded as JSON—including zero-win and "
      + "already-submitted runs—so keep the file before refreshing if online submission fails. "
      + "Anyone can drop that file into Replay to watch and seek through the run."],
  ],
  zh: [
    [`赢 ${MIN_SUBMITTABLE_WINS} 局即可上榜`, `人的每一个胜场都会计入。一次记录至少赢 `
      + `${MIN_SUBMITTABLE_WINS} 局即可提交。`],
    [`最多 ${LIMITS.maxRounds} 局 / ${LIMITS.maxFrames.toLocaleString()} 帧`,
      "排名分数是同一次连续排位记录里的累计胜场，达到任一上限就停止录制。"
      + "负场和双亡会一起展示。换迷宫或换对手会开始一次新记录。"],
    ["对手零延迟", "对手延迟必须是 0 帧。那是让智能体延迟出手的让步，排位不给任何让步。"],
    ["开局停顿不超过 0.5 秒", `默认的 ${RANKED_OPENING_DELAY_SECONDS} 秒或更短。`
      + "更短只会更难，所以允许。"],
    ["关闭瞬间转向", "开始排位时会自动关掉，并且必须保持关闭。"],
    ["三个榜分开排", "Hybrid、Laika 和 Killfield 各排各的。"],
    ["其余随意", "轮盘的前向区域、用触屏还是键盘、中途暂停思考——都不改变对局本身，所以都不限制。"],
    ["名字必填，账号可选", "每条记录都要有名字。是否显示 GitHub 账号由你决定，"
      + "一键提交里的账号只是自报的主页链接，不代表平台核验过账号归属。"],
    ["提交或保存录像", "大录像提交时会自动分块。任何已结束的排位记录都可下载为 JSON——"
      + "包括 0 胜和已提交记录。如果在线提交失败，请在刷新页面前保存文件。"
      + "任何人都可以把该文件拖入“录像”模式观看并拖动进度。"],
  ],
};

const COPY = {
  en: {
    htmlLang: "en",
    back: "Back to the game",
    heading: "Most wins",
    blurb: `Every win counts across a ranked run of up to ${LIMITS.maxRounds} rounds. `
      + "Every record here was replayed frame by frame before it landed.",
    rank: "#", player: "Player", score: "Wins", rounds: "Rounds",
    losses: "Losses", doubleKills: "Double KOs", when: "Verified",
    empty: `Nobody has submitted a win yet. The board is yours to open.`,
    note: "Runs are played at the default settings — no actuation delay for the "
      + "opponent — and scored by replaying the recorded inputs through the same "
      + "engine build. The score a submission claims is ignored.",
    unavailable: "The board could not be loaded.",
    rulesHeading: "What counts",
    langToggle: "中文",
    langAria: "Switch to Chinese",
  },
  zh: {
    htmlLang: "zh-Hans",
    back: "返回游戏",
    heading: "最多胜场",
    blurb: `一次排位最多记录 ${LIMITS.maxRounds} 局，每个胜场都计分。榜上每条记录都经过逐帧重放核验。`,
    rank: "#", player: "玩家", score: "胜", rounds: "总局数",
    losses: "负", doubleKills: "双亡", when: "核验于",
    empty: "还没有人提交胜场。第一个位置空着。",
    note: "成绩一律在默认设置下产生——对手不吃任何动作延迟——并由同一份引擎重放录制的输入重新计分，"
      + "提交时自报的分数不作数。",
    unavailable: "榜单加载失败。",
    rulesHeading: "上榜条件",
    langToggle: "EN",
    langAria: "切换到英文",
  },
};

const NODES = {
  back: "back-label", heading: "board-heading", blurb: "board-blurb",
  rank: "col-rank", player: "col-name", score: "col-score",
  rounds: "col-rounds", losses: "col-losses", doubleKills: "col-double-kills",
  when: "col-when", note: "board-note",
  rulesHeading: "rules-heading",
};

const tabs = [...document.querySelectorAll("#board-tabs .mode-btn")];
const body = document.getElementById("board-body");
const emptyNote = document.getElementById("board-empty");
const langToggle = document.getElementById("lang-toggle");
const rulesList = document.getElementById("rules-list");

let lang = loadLang();
let board = { entries: [] };
let failed = false;
let active = "hybrid";

const text = () => COPY[lang] ?? COPY.en;

function formatDate(iso) {
  const when = Date.parse(iso);
  if (!Number.isFinite(when)) return "—";
  return new Date(when).toLocaleDateString(lang === "zh" ? "zh-Hans" : "en-GB", {
    year: "numeric", month: "short", day: "numeric",
  });
}

function render() {
  const copy = text();
  document.documentElement.lang = copy.htmlLang;
  for (const [key, id] of Object.entries(NODES)) {
    document.getElementById(id).textContent = copy[key];
  }
  langToggle.textContent = copy.langToggle;
  langToggle.setAttribute("aria-label", copy.langAria);

  const rows = board.entries
    .filter((entry) => entry.board === active)
    .sort((a, b) => (b.wins ?? b.score) - (a.wins ?? a.score)
      || Date.parse(a.verifiedAt) - Date.parse(b.verifiedAt));

  body.replaceChildren();
  for (const [index, entry] of rows.entries()) {
    const row = document.createElement("tr");
    if (index === 0) row.className = "lead";
    const cells = [
      String(index + 1),
      null, // the name is built below, since it carries a link
      String(entry.wins ?? entry.score),
      String(entry.rounds),
      entry.losses == null ? "—" : String(entry.losses),
      entry.doubleKills == null ? "—" : String(entry.doubleKills),
      formatDate(entry.verifiedAt),
    ];
    cells.forEach((value, column) => {
      const cell = document.createElement("td");
      if (column === 1) {
        // Attacker-authored text: set as a text node, never parsed as markup.
        const name = document.createElement("span");
        name.className = "board-name";
        name.textContent = entry.name || "anonymous";
        cell.append(name);
        if (entry.github) {
          const handle = document.createElement("a");
          handle.className = "board-handle";
          handle.textContent = `@${entry.github}`;
          handle.rel = "noreferrer nofollow";
          handle.target = "_blank";
          handle.href = `https://github.com/${encodeURIComponent(entry.github)}`;
          cell.append(handle);
        }
      } else {
        cell.textContent = value;
      }
      row.append(cell);
    });
    body.append(row);
  }

  rulesList.replaceChildren();
  for (const [term, detail] of RULES[lang] ?? RULES.en) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = detail;
    rulesList.append(dt, dd);
  }

  emptyNote.hidden = rows.length > 0;
  emptyNote.textContent = failed ? copy.unavailable : copy.empty;
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.board === active));
}

tabs.forEach((tab) => tab.addEventListener("click", () => {
  active = tab.dataset.board;
  render();
}));

langToggle.addEventListener("click", () => {
  lang = lang === "en" ? "zh" : "en";
  saveLang(lang);
  render();
});

try {
  const response = await fetch("leaderboard.json", { cache: "no-store" });
  if (!response.ok) throw new Error(String(response.status));
  const loaded = await response.json();
  if (Array.isArray(loaded.entries)) board = loaded;
} catch {
  // The committed board exists even before the first record. A fetch or parse
  // failure is therefore different from a valid empty board and must not be
  // presented as "nobody has submitted yet".
  failed = true;
}
render();
