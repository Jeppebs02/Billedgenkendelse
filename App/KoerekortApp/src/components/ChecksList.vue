<template>
  <section class="card" style="padding:16px">
    <div class="row" style="justify-content:space-between;align-items:center">
      <h2 style="margin:0">Resultater</h2>
      <span class="badge">Score: {{ Math.round((score ?? 0) * 100) }}%</span>
    </div>
    <div class="hr"></div>

    <div class="list">
      <article v-for="c in checks" :key="c.key" class="list-item">
        <StatusIcon :status="c.status" />
        <div>
          <strong>{{ c.label }}</strong>
          <p class="text-muted" style="margin:.25rem 0 0">{{ c.message || fallbackMessage(c.status) }}</p>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup>
const props = defineProps({
  checks: { type: Array, default: ()=>[] },
  score: { type: Number, default: 0 },
})

function fallbackMessage(status){
  if(status==='pass') return 'Opfylder kravet.'
  if(status==='warn') return 'Næsten – overvej at tage et nyt billede.'
  return 'Opfylder ikke kravet.'
}
</script>

<script>
export default {
  components:{
    StatusIcon:{
      props:['status'],
      template:`
        <span :class="klass" aria-hidden="true">●</span>
      `,
      computed:{
        klass(){
          return {
            'text-good': this.status==='pass',
            'text-warn': this.status==='warn',
            'text-bad' : this.status==='fail'
          }
        }
      }
    }
  }
}
</script>
