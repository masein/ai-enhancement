// 21c: the runtime a report's published page was written for — exactly what
// the page uses, and nothing more (service/report_page.py serves it).
//   dash.data(id)        {status, data, columns}: "ok", "loading" or "error";
//                        a calc's id gives what its fn returned
//   dash.calc(id, spec)  {inputs, fn, title, description}: fn gets each input's rows
//   dash.onData(fn)      fn now, again when data arrives, and again on a resize
//   dash.params(), dash.setParams({...}), dash.resetParams(): setParams redraws
//   dash.setLink(name)   the URL's #name
//   dash.colors          the eight series colours, step4000's first
//   dash.refresh()       nothing
// Every dataset is read from our own server (data-api); nothing from another
// host. The shell's header line says when each source last changed, and the
// stats are read again when newer ones are worked out.
(function () {
  "use strict";
  var me = document.currentScript;
  var API = me.getAttribute("data-api");
  var IDS = JSON.parse(me.getAttribute("data-sets") || "[]");
  var DEFAULTS = JSON.parse(me.getAttribute("data-params") || "{}");
  var COLORS = ["#c0392b", "#2e86c1", "#8e44ad", "#d68910", "#16a085", "#7f8c8d", "#2c3e50",
    "#e67e22"];
  var sets = {}, calcs = {}, memo = {}, handlers = [];
  var version = 0;
  var params = Object.assign({}, DEFAULTS);
  IDS.forEach(function (id) { sets[id] = { status: "loading", data: [], columns: [] }; });

  function columns(rows) {
    var seen = [];
    (rows || []).forEach(function (r) {
      Object.keys(r || {}).forEach(function (k) { if (seen.indexOf(k) < 0) seen.push(k); });
    });
    return seen;
  }

  function data(id) {
    if (sets[id]) return sets[id];
    var c = calcs[id];
    if (!c) return { status: "error", data: [], columns: [] };
    if (memo[id] && memo[id].v === version) return memo[id].out;
    var ins = c.inputs.map(data), out;
    if (ins.some(function (x) { return x.status === "error"; }))
      out = { status: "error", data: [], columns: [] };
    else if (ins.some(function (x) { return x.status !== "ok"; }))
      out = { status: "loading", data: [], columns: [] };
    else {
      try {
        var rows = c.fn.apply(null, ins.map(function (x) { return x.data; })) || [];
        out = { status: "ok", data: rows, columns: columns(rows) };
      } catch (e) {
        console.error("dash: " + id + " couldn't be worked out", e);
        out = { status: "error", data: [], columns: [] };
      }
    }
    memo[id] = { v: version, out: out };
    return out;
  }

  function draw() {
    handlers.forEach(function (fn) {
      try { fn(); } catch (e) { console.error("dash: a section couldn't be drawn", e); }
    });
  }

  function set(id, status, rows) {
    sets[id] = { status: status, data: rows || [], columns: columns(rows) };
    version += 1;
    draw();
  }

  function sleep(ms) { return new Promise(function (ok) { setTimeout(ok, ms); }); }

  // a dataset from our server: 202 while its stats are being worked out —
  // asked again until they are in
  async function load(id, quiet) {
    for (;;) {
      var r;
      try {
        r = await fetch(API + "/" + encodeURIComponent(id), { cache: "no-store",
          headers: { Accept: "application/json" } });
      } catch (e) {
        if (!quiet) set(id, "error");
        return;
      }
      if (r.status === 202) { await sleep(5000); continue; }
      if (!r.ok) { if (!quiet) set(id, "error"); return; }
      var rows;
      try { rows = await r.json(); } catch (e) { if (!quiet) set(id, "error"); return; }
      set(id, "ok", Array.isArray(rows) ? rows : []);
      return;
    }
  }

  window.dash = {
    data: data,
    calc: function (id, spec) {
      calcs[id] = { inputs: (spec && spec.inputs) || [], fn: spec.fn, title: spec.title || "",
        description: spec.description || "" };
      delete memo[id];
    },
    onData: function (fn) {
      handlers.push(fn);
      try { fn(); } catch (e) { console.error("dash: a section couldn't be drawn", e); }
    },
    params: function () { return Object.assign({}, params); },
    setParams: function (p) { params = Object.assign({}, params, p || {}); version += 1; draw(); },
    resetParams: function () { params = Object.assign({}, DEFAULTS); version += 1; draw(); },
    setLink: function (name) {
      try { history.replaceState(history.state, "", "#" + encodeURIComponent(name)); }
      catch (e) { location.hash = name; }
    },
    colors: COLORS.slice(),
    refresh: function () {}
  };

  // drawn again when the width changes (a phone's bar showing or hiding
  // changes only the height)
  var raf = 0, width = window.innerWidth;
  window.addEventListener("resize", function () {
    if (window.innerWidth === width) return;
    width = window.innerWidth;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(draw);
  });

  // the board's theme, changed in another tab
  window.addEventListener("storage", function (e) {
    if (e.key !== "bench-theme") return;
    if (!e.newValue || e.newValue === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", e.newValue);
    draw();
  });

  IDS.forEach(function (id) { load(id); });

  // the shell's header line: when each source last changed — read again, and
  // the datasets a newer source changed read again with it
  var seen = null;
  async function fresh() {
    var wait = 60000;
    try {
      var r = await fetch(API, { cache: "no-store", headers: { Accept: "application/json" } });
      if (r.ok) {
        var info = await r.json();
        var line = document.querySelector("[data-dash-fresh]");
        if (line) {
          line.replaceChildren();
          info.sources.forEach(function (s, i) {
            var span = document.createElement("span");
            span.setAttribute("data-dash-source", s.key);
            span.textContent = s.words;
            line.appendChild(span);
          });
          if (info.credit && info.credit.url) {
            var a = document.createElement("a");
            a.href = info.credit.url; a.target = "_blank"; a.rel = "noopener";
            a.textContent = info.credit.words;
            line.appendChild(a);
          }
        }
        var at = {};
        info.sources.forEach(function (s) { at[s.key] = s.at; });
        if (seen) {
          if (at.ours !== seen.ours) { load("ours", true); load("tokens", true); }
          if (at.epoch !== seen.epoch || at.cards !== seen.cards) load("others", true);
          if (at.stats !== seen.stats) { load("tokens", true); load("loops", true); }
        }
        seen = at;
        if (info.working && info.working.todo) wait = 5000;
      }
    } catch (e) { /* asked again in a minute */ }
    setTimeout(fresh, wait);
  }
  fresh();
})();
