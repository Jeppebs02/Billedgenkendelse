import { defineStore } from 'pinia'
import { ref, computed } from 'vue'


export const useDataStore = defineStore('data', () => {

  const jsonResult = ref(null)
  const imageSrc = ref(null)
  const getImageSrc = computed(() => imageSrc.value )
  const getJsonResult = computed(() => jsonResult.value)

  function setImageSrc(newImage) {imageSrc.value = newImage}
  function setJsonResult(newJson) {jsonResult.value = newJson}



  return {imageSrc, jsonResult, getImageSrc, getJsonResult, setImageSrc, setJsonResult}
})
