/* Docs runtime: build the on-page table of contents and highlight as you read. */

const article = document.querySelector(".article");
const toc = document.querySelector(".toc");

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}

if (article && toc) {
  const headings = [...article.querySelectorAll("h2, h3")];
  const used = new Set();

  for (const heading of headings) {
    if (!heading.id) {
      let id = slugify(heading.textContent);
      let n = 2;
      while (used.has(id)) id = `${slugify(heading.textContent)}-${n++}`;
      heading.id = id;
    }
    used.add(heading.id);
  }

  if (headings.length > 1) {
    const links = headings
      .map(
        (h) =>
          `<a href="#${h.id}" class="${h.tagName === "H3" ? "level-3" : ""}">${h.textContent}</a>`
      )
      .join("");
    toc.innerHTML = `<p class="toc-title">On this page</p>${links}`;

    const anchors = new Map(
      [...toc.querySelectorAll("a")].map((a) => [a.getAttribute("href").slice(1), a])
    );

    // Highlight the heading nearest the top of the viewport.
    const seen = new Set();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) seen.add(entry.target.id);
          else seen.delete(entry.target.id);
        }
        const current = headings.find((h) => seen.has(h.id));
        for (const [id, anchor] of anchors) {
          anchor.classList.toggle("is-active", !!current && id === current.id);
        }
      },
      { rootMargin: "0px 0px -70% 0px", threshold: 0 }
    );
    headings.forEach((h) => observer.observe(h));
  } else {
    toc.remove();
  }
}

// Mark the current page in the sidebar without hardcoding it per file.
const here = location.pathname.split("/").pop() || "index.html";
for (const link of document.querySelectorAll(".side-links a")) {
  const target = link.getAttribute("href");
  if (target === here || (here === "" && target === "index.html")) {
    link.setAttribute("aria-current", "page");
  }
}
