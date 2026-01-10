/**
 * Star Genius - Internationalization (i18n) Module
 */

const i18n = {
    translations: {},
    currentLang: 'en',

    async init() {
        // Load translations
        try {
            const response = await fetch('/translations.json');
            this.translations = await response.json();
        } catch (e) {
            console.error('Failed to load translations:', e);
            return;
        }

        // Get saved language or detect from browser
        const saved = localStorage.getItem('sg-lang');
        const browserLang = navigator.language.split('-')[0];
        this.currentLang = saved || (this.translations[browserLang] ? browserLang : 'en');

        // Apply translations
        this.applyTranslations();

        // Setup language selector
        this.setupSelector();
    },

    t(key) {
        return this.translations[this.currentLang]?.[key] ||
            this.translations['en']?.[key] ||
            key;
    },

    setLanguage(lang) {
        if (!this.translations[lang]) return;
        this.currentLang = lang;
        localStorage.setItem('sg-lang', lang);
        this.applyTranslations();

        // Update selector
        const select = document.getElementById('lang-select');
        if (select) select.value = lang;
    },

    applyTranslations() {
        // Update all elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            el.textContent = this.t(key);
        });

        // Update elements with data-i18n-placeholder
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.placeholder = this.t(key);
        });

        // Update elements with data-i18n-title
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            el.title = this.t(key);
        });
    },

    setupSelector() {
        const select = document.getElementById('lang-select');
        if (!select) return;

        select.value = this.currentLang;
        select.addEventListener('change', (e) => {
            this.setLanguage(e.target.value);
        });
    }
};

// Export to window
window.i18n = i18n;
