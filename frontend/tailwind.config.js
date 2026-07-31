/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // ── Primary ────────────────────────────────────────────
        'primary':                 '#efffff',
        'primary-container':       '#00f7ff',
        'primary-fixed':           '#54f8ff',
        'primary-fixed-dim':       '#00dce3',
        'on-primary':              '#003739',
        'on-primary-container':    '#006d71',
        'on-primary-fixed':        '#002021',
        'on-primary-fixed-variant':'#004f52',
        'inverse-primary':         '#00696d',
        // ── Secondary ──────────────────────────────────────────
        'secondary':                   '#ffdb9f',
        'secondary-container':         '#ffb700',
        'secondary-fixed':             '#ffdea9',
        'secondary-fixed-dim':         '#ffba26',
        'on-secondary':                '#422d00',
        'on-secondary-container':      '#6b4b00',
        'on-secondary-fixed':          '#271900',
        'on-secondary-fixed-variant':  '#5e4100',
        // ── Tertiary ───────────────────────────────────────────
        'tertiary':                    '#fffaff',
        'tertiary-container':          '#ffde41',
        'tertiary-fixed':              '#ffe260',
        'tertiary-fixed-dim':          '#e4c526',
        'on-tertiary':                 '#3a3000',
        'on-tertiary-container':       '#736100',
        'on-tertiary-fixed':           '#221b00',
        'on-tertiary-fixed-variant':   '#534600',
        // ── Error / Critical ───────────────────────────────────
        'error':              '#ffb4ab',
        'error-container':    '#93000a',
        'on-error':           '#690005',
        'on-error-container': '#ffdad6',
        // ── Surface ────────────────────────────────────────────
        'surface':                  '#131313',
        'surface-dim':              '#131313',
        'surface-bright':           '#3a3939',
        'surface-container-lowest': '#0e0e0e',
        'surface-container-low':    '#1c1b1b',
        'surface-container':        '#201f1f',
        'surface-container-high':   '#2a2a2a',
        'surface-container-highest':'#353534',
        'surface-variant':          '#353534',
        'surface-tint':             '#00dce3',
        'on-surface':               '#e5e2e1',
        'on-surface-variant':       '#b9caca',
        // ── Background ─────────────────────────────────────────
        'background':    '#131313',
        'on-background': '#e5e2e1',
        // ── Outline ────────────────────────────────────────────
        'outline':         '#849494',
        'outline-variant': '#3a4a4a',
        // ── Inverse ────────────────────────────────────────────
        'inverse-surface':    '#e5e2e1',
        'inverse-on-surface': '#313030',
      },

      fontFamily: {
        headline: ['"Space Grotesk"', 'sans-serif'],
        body:     ['Inter', 'sans-serif'],
        label:    ['"Space Grotesk"', 'sans-serif'],
        mono:     ['"Courier New"', 'Courier', 'monospace'],
      },

      fontSize: {
        '8':  ['8px',  { lineHeight: '1' }],
        '9':  ['9px',  { lineHeight: '1' }],
        '10': ['10px', { lineHeight: '1.2' }],
        '11': ['11px', { lineHeight: '1.3' }],
      },

      letterSpacing: {
        tightest: '-0.05em',
        wider:    '0.05em',
        widest:   '0.1em',
        ultra:    '0.2em',
        max:      '0.3em',
      },

      borderRadius: {
        DEFAULT: '0.125rem',
        lg:      '0.25rem',
        xl:      '0.5rem',
        full:    '0.75rem',
      },

      keyframes: {
        shimmer: {
          '0%':   { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition:  '1000px 0' },
        },
        'glitch-pulse': {
          '0%, 100%': { transform: 'translate(0)', opacity: '1' },
          '10%':      { transform: 'translate(-2px, -1px)', opacity: '0.9' },
          '20%':      { transform: 'translate(2px, 1px)',   opacity: '0.95' },
          '30%':      { transform: 'translate(-1px, 2px)' },
          '40%':      { transform: 'translate(1px, -2px)' },
          '50%':      { transform: 'translate(-2px, 1px)' },
        },
        'marquee-scroll': {
          '0%':   { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        shockwave: {
          '0%':   { transform: 'scale(1)', opacity: '0.8' },
          '100%': { transform: 'scale(3)', opacity: '0' },
        },
        'loading-bar': {
          '0%':   { left: '-100%' },
          '100%': { left: '100%' },
        },
        'radar-sweep': {
          '0%':   { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
      },

      animation: {
        shimmer:        'shimmer 2s ease-in-out infinite',
        'glitch-pulse': 'glitch-pulse 0.3s ease-in-out',
        'marquee-scroll':'marquee-scroll 30s linear infinite',
        shockwave:      'shockwave 1.5s ease-out infinite',
        'loading-bar':  'loading-bar 2s infinite linear',
        'radar-sweep':  'radar-sweep 4s linear infinite',
      },

      boxShadow: {
        'glow-cyan':    '0 0 15px rgba(0, 247, 255, 0.4)',
        'glow-cyan-lg': '0 0 30px rgba(0, 247, 255, 0.3)',
        'glow-amber':   '0 0 15px rgba(255, 183, 0, 0.4)',
        'glow-error':   '0 0 15px rgba(147, 0, 10, 0.5)',
        'glow-text':    '0 0 8px rgba(0, 247, 255, 0.5)',
      },
    },
  },
  plugins: [],
}
