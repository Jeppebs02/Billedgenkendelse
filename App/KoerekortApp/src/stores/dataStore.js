import { defineStore } from 'pinia'
import { ref, computed } from 'vue'


export const useDataStore = defineStore('data', () => {

  const jsonResult = ref(null)
  const imageSrc = ref(null)
  const getImageSrc = computed(() => imageSrc.value )
  const getJsonResult = computed(() => jsonResult.value)

  function setImageSrc(newImage) {imageSrc.value = newImage}
  function setJsonResult(newJson) {jsonResult.value = newJson}

  const version = ref('')
  const getVersion = computed(() => version.value)

  async function fetchVersion() {
    if (version.value) return
    try {
      const response = await fetch('https://api.terragrouplabs.net', {
        method: 'GET',
        headers: {
          'x-api-key': 'a541fe33-6c48-490c-b71a-eadab16594de'
        },
      })
      const text = await response.text()
      // Replace p tags
      version.value = text.replace(/<[^>]*>/g, '').replace('OK. App version: ', '').trim()
    } catch (error) {
      console.error('Failed to fetch version:', error)
    }
  }

  return {imageSrc, jsonResult, getImageSrc, getJsonResult, setImageSrc, setJsonResult, version, getVersion, fetchVersion}
})
