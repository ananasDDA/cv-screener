/* Renders a full candidate profile (get_candidate result) into a container. Shared by both widgets. */
function fmtMonth(v) {
  if (!v) return "Present";
  const [y, m] = v.split("-");
  return new Date(Number(y), Number(m) - 1, 1).toLocaleString("en", { month: "short", year: "numeric" });
}
function renderProfile(el, p, opts = {}) {
  const langs = (p.languages || []).map((l) => `<span class="chip">${esc(l.name)} <span class="faint">${esc(l.level)}</span></span>`).join("");
  const skills = (p.skills || []).map((s) => `<span class="chip accent">${esc(s)}</span>`).join("");
  const exp = (p.experience || []).map((e) => `
    <div class="job">
      <div class="job-head"><b>${esc(e.title)}</b> <span class="muted">· ${esc(e.company)}, ${esc(e.location)}</span>
        <span class="faint mono">${fmtMonth(e.start)} – ${fmtMonth(e.end)}</span></div>
      <ul>${(e.highlights || []).map((h) => `<li>${esc(h)}</li>`).join("")}</ul>
    </div>`).join("");
  const edu = (p.education || []).map((e) => `<div><b>${esc(e.degree)} ${esc(e.field)}</b> <span class="muted">· ${esc(e.institution)}, ${e.start_year}–${e.end_year}</span></div>`).join("");
  const certs = (p.certifications || []).map((c) => `<li>${esc(c)}</li>`).join("");
  el.innerHTML = `
    <div class="profile">
      <div class="head">
        <div class="avatar big" data-photo="${esc(p.id)}">${initials(p.full_name)}</div>
        <div class="who">
          <div class="name">${esc(p.full_name)}</div>
          <div class="headline">${esc(p.headline)}</div>
          <div class="muted small">${esc(p.seniority)} · ${esc(p.role_family)} · ${p.years_experience} yrs · ${esc(p.city)}, ${esc(p.country)}</div>
          <div class="faint small mono">${esc(p.email)} · ${esc(p.phone)}${p.linkedin ? " · " + esc(p.linkedin) : ""}${p.github ? " · " + esc(p.github) : ""}</div>
        </div>
        ${opts.actions || ""}
      </div>
      <p class="summary">${esc(p.summary)}</p>
      <div class="section"><h4>Skills</h4>${skills}</div>
      <div class="section"><h4>Experience</h4>${exp}</div>
      <div class="cols">
        <div class="section"><h4>Education</h4>${edu}</div>
        <div class="section"><h4>Languages</h4>${langs}${certs ? `<h4>Certifications</h4><ul>${certs}</ul>` : ""}</div>
      </div>
    </div>`;
  const av = el.querySelector(".avatar.big");
  if (opts.photo) {
    opts.photo(p.id).then((uri) => { if (uri) av.outerHTML = `<img class="avatar big" src="${uri}" alt="">`; }).catch(() => {});
  }
}
const PROFILE_CSS = `
  .profile { padding: 16px 18px; }
  .head { display: flex; gap: 14px; align-items: flex-start; }
  .avatar.big { width: 64px; height: 64px; font-size: var(--font-size-l); border-radius: var(--radius-m); }
  .who { flex: 1; min-width: 0; }
  .name { font-size: var(--font-size-l); font-weight: 600; letter-spacing: -0.3px; }
  .headline { color: var(--color-accent); font-weight: 500; }
  .small { font-size: var(--font-size-s); }
  .summary { margin: 12px 0 4px; }
  .section { margin-top: 12px; }
  h4 { margin: 0 0 6px; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--color-text-secondary); }
  .job { margin-bottom: 10px; }
  .job-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; }
  .job-head .mono { margin-left: auto; }
  .job ul, .section ul { margin: 4px 0 0; padding-left: 18px; }
  .job li { margin-bottom: 2px; font-size: var(--font-size-s); }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 560px) { .cols { grid-template-columns: 1fr; } }
`;
