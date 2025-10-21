<template>
  <section class="card" style="padding:16px">
    <div class="row" style="justify-content:space-between;align-items:center">
      <div>
        <h2 style="margin:0">Opsummering</h2>
        <p class="text-muted" style="margin:.25rem 0 0">{{ summary }}</p>
      </div>
      <div class="row" style="gap:8px">
        <a v-if="reportUrl" class="btn" :href="reportUrl" target="_blank" rel="noopener">Åbn rapport</a>
        <button class="btn btn-primary" @click="$emit('retry')">Prøv igen</button>
      </div>
    </div>
  </section>
</template>

<script setup>
defineProps({
  summary: { type:String, default:'' },
  reportUrl: { type:String, default:'' },
})
defineEmits(['retry'])
</script>
