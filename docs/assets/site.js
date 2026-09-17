/* CV Screener landing — theme toggle, copy buttons and the live coverflow deck.
   The deck's geometry and paint loop are ported from src/cv_screener/widgets/deck.html
   (the project's own widget); here they run on the static assets/deck.js payload instead
   of MCP calls, and photos come from assets/photos/<id>.jpg. */
(function () {
  "use strict";

  /* ------------------------------------------------------------------ theme */
  var root = document.documentElement;
  var KEY = "cv-screener-theme";
  var stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) { /* private mode */ }
  if (stored === "dark") root.classList.add("theme-dark");
  if (stored === "light") root.classList.add("theme-light");

  function prefersDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function isDark() {
    if (root.classList.contains("theme-dark")) return true;
    if (root.classList.contains("theme-light")) return false;
    return prefersDark();
  }
  var toggle = document.getElementById("theme-toggle");
  function syncToggle() {
    if (toggle) toggle.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
  }
  syncToggle();
  if (toggle) {
    toggle.addEventListener("click", function () {
      var dark = !isDark();
      root.classList.toggle("theme-dark", dark);
      root.classList.toggle("theme-light", !dark);
      try { localStorage.setItem(KEY, dark ? "dark" : "light"); } catch (e) { /* ignore */ }
      syncToggle();
    });
  }

  /* ------------------------------------------------------------- copy blocks */
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    var label = btn.textContent;
    btn.addEventListener("click", function () {
      var pre = document.getElementById(btn.getAttribute("data-copy"));
      if (!pre) return;
      var text = pre.textContent;
      var done = function () {
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = label; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { select(pre); });
      } else {
        select(pre);
      }
    });
  });
  function select(pre) {
    var r = document.createRange();
    r.selectNodeContents(pre);
    var s = window.getSelection();
    s.removeAllRanges();
    s.addRange(r);
  }

  /* -------------------------------------------------------------------- deck */
  var track = document.getElementById("track");
  var stage = document.getElementById("stage");
  if (!track || !stage) return;

  var flat = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (flat) track.classList.add("no3d");

  var people = [], N = 0, slots = [], cards = [], looped = false;
  var slotW = 0, base = 0, viewW = 0, lit = -1;
  var was = [], geo = [];

  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function initials(name) {
    return String(name || "").split(/\s+/).filter(Boolean).slice(0, 2)
      .map(function (w) { return w[0].toUpperCase(); }).join("");
  }
  function rgba(hex, a) {
    var n = parseInt(String(hex).slice(1), 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }

  function build(list) {
    people = list;
    N = list.length;
    var pads = track.querySelectorAll(".pad");
    var copies = N > 2 ? 3 : 1; // three copies keep a card on either side, as in the widget
    looped = copies > 1;
    for (var c = 0; c < copies; c++) {
      list.forEach(function (p, i) {
        var slot = document.createElement("div");
        slot.className = "slot";
        slot.innerHTML =
          '<div class="poster" data-i="' + i + '" style="--accent:' + esc(p.accent || "#16324f") + '">' +
          '<div class="mono-initials">' + initials(p.name) + "</div>" +
          '<img class="photo" src="assets/photos/' + esc(p.id) + '.jpg" alt="" loading="lazy" decoding="async">' +
          '<div class="shade"></div>' +
          '<div class="mark"><span>' + esc(p.role_family) + " · " + esc(p.seniority) + "</span></div>" +
          '<div class="cap"><div class="pname">' + esc(p.name) + "</div>" +
          '<div class="phead">' + esc(p.headline) + "</div></div></div>";
        track.insertBefore(slot, pads[1]);
      });
    }
    slots = [].slice.call(track.querySelectorAll(".slot"));
    cards = slots.map(function (s) { return s.firstElementChild; });
    cards.forEach(function (card, k) {
      card.addEventListener("click", function () { centre(k); });
      var img = card.querySelector(".photo");
      var on = function () { img.classList.add("on"); };
      if (img.complete && img.naturalWidth) on(); else img.addEventListener("load", on);
    });
    document.getElementById("count").innerHTML =
      "<b>" + N + "</b> candidate" + (N === 1 ? "" : "s") + " in the collection";
    reset(false);
  }

  function measure() {
    viewW = track.clientWidth;
    var w = Math.max(190, Math.min(260, Math.round(viewW * 0.42)));
    track.style.setProperty("--slot", w + "px");
    slotW = slots[0] ? slots[0].getBoundingClientRect().width : 0;
    base = slotW * N;
    var p = Math.max(0, (viewW - slotW) / 2);
    track.querySelectorAll(".pad").forEach(function (el) { el.style.width = p + "px"; });
    geo.length = 0;
    slots.forEach(function (s) { geo.push(s.offsetLeft + s.offsetWidth / 2); });
  }

  function paint() {
    var vc = track.scrollLeft + viewW / 2;
    var near = 0, nd = Infinity;
    for (var i = 0; i < cards.length; i++) {
      var t = (geo[i] - vc) / slotW, at = Math.abs(t);
      if (at < nd) { nd = at; near = i; }
      var ct = Math.max(-1.5, Math.min(1.5, t)), act = Math.abs(ct);
      var scale = Math.round((1.13 - Math.min(act, 1.4) * 0.19) * 500) / 500;
      var op = Math.round((1 - Math.min(at, 1.9) * 0.42) * 100) / 100;
      var rot = flat ? 0 : Math.round(-18 * ct * 4) / 4;
      var z = 200 - Math.round(at * 20);
      var w = was[i] || (was[i] = {});
      if (w.s !== scale || w.r !== rot) {
        w.s = scale; w.r = rot;
        cards[i].style.transform = flat
          ? "scale(" + scale + ")"
          : "perspective(1150px) scale(" + scale + ") rotateY(" + rot + "deg)";
      }
      if (w.o !== op) { w.o = op; cards[i].style.opacity = op; }
      if (w.z !== z) { w.z = z; cards[i].style.zIndex = z; }
    }
    var c = N ? near % N : -1;
    if (c !== lit && c >= 0) { lit = c; showInfo(people[c]); }
  }

  function rewind() {
    if (!looped || !base) return;
    var x = track.scrollLeft;
    if (x >= base * 2) track.scrollLeft = x - base;
    else if (x < base) track.scrollLeft = x + base;
  }

  var queued = false, idle = null;
  track.addEventListener("scroll", function () {
    if (!queued) {
      queued = true;
      requestAnimationFrame(function () { queued = false; paint(); });
    }
    clearTimeout(idle);
    idle = setTimeout(function () { rewind(); paint(); }, 180);
  }, { passive: true });

  function focused() {
    var vc = track.scrollLeft + track.clientWidth / 2, best = 0, bd = Infinity;
    slots.forEach(function (s, i) {
      var d = Math.abs(s.offsetLeft + s.offsetWidth / 2 - vc);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }
  function centre(k, smooth) {
    track.scrollTo({ left: geo[k] - viewW / 2, behavior: smooth !== false && !flat ? "smooth" : "auto" });
  }
  function step(d) { centre(Math.max(0, Math.min(slots.length - 1, focused() + d))); }
  function reset(keep) {
    var k = keep === false ? -1 : focused();
    measure();
    if (k >= 0 && geo[k] != null) track.scrollLeft = geo[k] - viewW / 2;
    else if (looped) track.scrollLeft = base;
    paint();
  }

  document.getElementById("prev").addEventListener("click", function () { step(-1); });
  document.getElementById("next").addEventListener("click", function () { step(1); });
  /* Arrow keys steer the deck only while it has focus, and the vertical wheel is never
     hijacked — the page keeps scrolling normally. */
  document.getElementById("deckview").addEventListener("keydown", function (e) {
    if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
  });

  var lastW = 0;
  if (window.ResizeObserver) {
    new ResizeObserver(function () {
      if (track.clientWidth !== lastW) {
        lastW = track.clientWidth;
        if (slots.length) reset();
      }
    }).observe(track);
  } else {
    window.addEventListener("resize", function () { if (slots.length) reset(); });
  }

  function showInfo(p) {
    stage.style.setProperty("--glow", rgba(p.accent || "#0081ff", 0.22));
    document.getElementById("info").innerHTML =
      '<div class="line"><span class="title">' + esc(p.name) + "</span>" +
      '<span class="chip">' + esc(p.seniority) + "</span>" +
      '<span class="chip">' + esc(p.role_family) + "</span>" +
      '<span class="chip">' + esc(p.years_experience) + " yrs</span>" +
      (p.country ? '<span class="chip">' + esc(p.country) + "</span>" : "") +
      "</div>" +
      '<div class="muted langs">' + esc(p.languages) + "</div>" +
      '<div class="skills">' + (p.top_skills || []).map(function (s) {
        return '<span class="chip accent">' + esc(s) + "</span>";
      }).join("") + "</div>";
  }

  var data = window.CV_DECK;
  if (Array.isArray(data) && data.length) {
    build(data);
  } else {
    document.getElementById("count").textContent = "Could not load the collection.";
  }
})();
