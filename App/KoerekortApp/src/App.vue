<template>
  <div class="min-h-screen bg-surface text-ink">
    <header class="container py-4 flex items-center justify-between">
      <h1 class="text-xl font-semibold">Kørekortbillede • Validator</h1>
      <nav class="text-sm opacity-70">SPA • Vue 3</nav>
    </header>
    <main class="container py-6">
      <RouterView />
    </main>
    <footer class="container py-8 text-xs opacity-70">
      <p>Tip: Træk & slip billedet ind i feltet, eller brug kameraet på mobilen.</p>
    </footer>
  </div>
</template>

<script setup>
// no-op
</script>
