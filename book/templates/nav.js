class NavigationDropdown extends HTMLElement {
    constructor() {
      super();

      // Get the initial expanded state from the attribute, default to false
      const initialExpanded = this.getAttribute('expanded') === 'true';

      // Build chapter URL prefix relative to current page location.
      // On chapter pages (served from /c/), use sibling paths; otherwise prefix with c/.
      const inChaptersDir = window.location.pathname.includes('/c/');
      const p = inChaptersDir ? '' : 'c/';

      this.innerHTML = `
        <div>
          <button class="dropdown-button" aria-expanded="${initialExpanded}">
            <span><strong>Navigation</strong></span>
            <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M19 9l-7 7-7-7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>

          <div class="dropdown-content${initialExpanded ? ' open' : ''}">
    <nav class="chapter-nav">

      <div class="section">
        <h3>Defining HCLLMs</h3>
        <ol start="1">
          <li><a href="${p}01-introduction.html">Introduction</a></li>
          <li><a href="${p}02-hci.html">HCI for HCLLMs</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Developing HCLLMs</h3>
        <ol start="3">
          <li><a href="${p}03-data.html">Data Pipeline</a></li>
          <li><a href="${p}04-nlp.html">NLP for HCLLMs</a></li>
          <li><a href="${p}05-eval.html">Evaluations</a></li>
        </ol>
      </div>

      <div class="section">
        <h3>Deploying HCLLMs and Case Study</h3>
        <ol start="6">
          <li><a href="${p}06-responsible.html">Responsible HCLLMs</a></li>
          <li><a href="${p}07-casestudy.html">Case Study: HCLLMs and the Future of Work</a></li>
        </ol>
      </div>
    </nav>
    <div id="search"></div>
  </div>
</div>
      `;

      // Initialize Pagefind search if available
      var searchEl = this.querySelector('#search');
      if (searchEl && typeof PagefindUI !== 'undefined') {
        new PagefindUI({ element: searchEl, showImages: false });
      }

      // Set up click handler
      const button = this.querySelector('.dropdown-button');
      const content = this.querySelector('.dropdown-content');

      button.addEventListener('click', () => {
        const isExpanded = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', !isExpanded);
        content.classList.toggle('open');
      });
    }

    // Add attribute change observer
    static get observedAttributes() {
      return ['expanded'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
      if (name === 'expanded') {
        const button = this.querySelector('.dropdown-button');
        const content = this.querySelector('.dropdown-content');
        const isExpanded = newValue === 'true';

        if (button && content) {
          button.setAttribute('aria-expanded', isExpanded);
          content.classList.toggle('open', isExpanded);
        }
      }
    }
}

// Only define the component once
if (!customElements.get('navigation-dropdown')) {
  customElements.define('navigation-dropdown', NavigationDropdown);
}
