// Enhanced Landing Page Component with Visual Effects
// Add these className additions to existing elements in LandingPage.jsx

/*
USAGE GUIDE - Add these classes to existing elements:

1. HERO SECTION:
   - Add "neon-glow" to main heading
   - Add "shimmer-text" to subtitle
   - Add "magnetic-button hover-lift" to CTA buttons

2. AGENT CARDS:
   - Add "hover-lift stagger-item" to each card
   - Add "holographic" to agent names

3. DEMO SECTION:
   - Add "glitch" to demo trigger button
   - Add "fade-in-up" to scanner feed items

4. PRICING CARDS:
   - Add "hover-lift" to pricing cards
   - Add "glow-pulse" to popular badge

5. FOOTER:
   - Add "ripple-effect" to social icons

EXAMPLE MODIFICATIONS:
*/

// Before:
// <h1 className="text-6xl md:text-8xl font-extrabold">

// After:
// <h1 className="text-6xl md:text-8xl font-extrabold neon-glow">

// Before:
// <button onClick={...} className="px-8 py-4 bg-white">

// After:
// <button onClick={...} className="px-8 py-4 bg-white magnetic-button hover-lift">

/*
PARTICLE BACKGROUND - Add to hero section:

<div className="absolute inset-0 overflow-hidden pointer-events-none">
  <div className="particle" style={{ left: '10%' }}></div>
  <div className="particle" style={{ left: '30%' }}></div>
  <div className="particle" style={{ left: '50%' }}></div>
  <div className="particle" style={{ left: '70%' }}></div>
  <div className="particle" style={{ left: '90%' }}></div>
</div>
*/

/*
FLOATING BACKGROUND GLOWS - Replace existing background divs:

<div className="fixed inset-0 pointer-events-none z-0">
  <div 
    className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px]"
    style={{ animation: 'float-slow 20s infinite ease-in-out' }}
  />
  <div 
    className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]"
    style={{ animation: 'float-medium 15s infinite ease-in-out' }}
  />
</div>
*/

export const visualEffectsGuide = {
    hero: {
        heading: "neon-glow gradient-text",
        subtitle: "shimmer-text",
        buttons: "magnetic-button hover-lift ripple-effect",
        background: "Add particle divs"
    },
    agents: {
        cards: "hover-lift stagger-item",
        titles: "holographic",
        icons: "Add scale-110 on hover"
    },
    demo: {
        button: "glitch magnetic-button",
        feed: "fade-in-up",
        chart: "Add smooth transitions"
    },
    pricing: {
        cards: "hover-lift",
        popular: "glow-pulse",
        buttons: "magnetic-button ripple-effect"
    },
    footer: {
        social: "ripple-effect hover-lift",
        links: "Add hover:text-red-500 transition"
    }
};
