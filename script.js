document.addEventListener('DOMContentLoaded', function () {
    if (location.protocol === 'file:') {
        const w = document.getElementById('local-file-warning');
        if (w) {
            w.hidden = false;
        }
    }

    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start',
                });
            }
        });
    });

    initTeaserAutoplaySync();
});

/**
 * Muted autoplay groups: keep all videos in a .sync-group--autoplay aligned in time.
 */
function initTeaserAutoplaySync() {
    document.querySelectorAll('.sync-group--autoplay').forEach((container) => {
        const videos = Array.from(container.querySelectorAll('video'));
        if (videos.length === 0) {
            return;
        }

        videos.forEach((v) => {
            v.controls = false;
            v.setAttribute('playsinline', '');
            v.muted = true;
        });

        if (videos.length === 1) {
            videos[0].play().catch(() => {});
            return;
        }

        function pickMaster() {
            let best = videos[0];
            let bestDur = best.duration && !Number.isNaN(best.duration) ? best.duration : Infinity;
            videos.forEach((v) => {
                const d = v.duration && !Number.isNaN(v.duration) ? v.duration : Infinity;
                if (d < bestDur) {
                    bestDur = d;
                    best = v;
                }
            });
            return best;
        }

        let master = videos[0];
        let syncing = false;

        function syncFollowers() {
            if (syncing) {
                return;
            }
            const t = master.currentTime;
            syncing = true;
            requestAnimationFrame(() => {
                videos.forEach((v) => {
                    if (v === master) {
                        return;
                    }
                    if (Math.abs(v.currentTime - t) > 0.12) {
                        try {
                            v.currentTime = t;
                        } catch (e) {
                            /* ignore */
                        }
                    }
                });
                syncing = false;
            });
        }

        function attachMasterListeners() {
            master.addEventListener('timeupdate', syncFollowers);
            master.addEventListener('seeked', syncFollowers);
        }

        Promise.all(
            videos.map(
                (v) =>
                    new Promise((resolve) => {
                        if (v.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                            resolve();
                        } else {
                            v.addEventListener('loadeddata', () => resolve(), { once: true });
                            v.addEventListener('error', () => resolve(), { once: true });
                        }
                    })
            )
        ).then(() => {
            master = pickMaster();
            attachMasterListeners();
            videos.forEach((v) => {
                v.play().catch(() => {});
            });
        });
    });
}

function copyBibtex() {
    const codeEl = document.querySelector('.bibtex-box pre code');
    if (!codeEl) {
        alert('BibTeX block not found.');
        return;
    }
    const bibtexText = codeEl.textContent.trim();

    navigator.clipboard.writeText(bibtexText).then(() => {
        const btn = document.querySelector('.copy-btn');
        const originalText = btn.textContent;
        btn.textContent = '✓ Copied!';
        btn.style.background = '#27ae60';

        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.background = '#4a90e2';
        }, 2000);
    }).catch(() => {
        alert('Failed to copy BibTeX. Please copy manually.');
    });
}
