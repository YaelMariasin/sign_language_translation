<template>
  <div id="app">
    <div class="content" 
         :style="{ height: result ? auto : auto }"
    >
      <h1>Sign Language Video Translator</h1>
      <video-upload @video-uploaded="handleVideoUploaded" />
      <translation-result :result="result" v-if="result"/>
    </div>
  </div>
</template>

<script>
import VideoUpload from './components/VideoUpload.vue';
import TranslationResult from './components/TranslationResult.vue';

export default {
  name: 'App',
  components: {
    VideoUpload,
    TranslationResult
  },
  data() {
    return {
      result: null
    };
  },
  methods: {
    handleVideoUploaded(result) {
      this.result = result;
    }
  }
};
</script>

<style scoped>

#app {
  text-align: center;
  font-family: 'Roboto', sans-serif;
  color: #333;
  background: 
    url('@/assets/sign1.jpg') left center no-repeat, 
    url('@/assets/sign2.jpg') center center no-repeat, 
    url('@/assets/sign3.jpg') right center no-repeat;
  background-size: 33.33vw 100vh;
  background-repeat: no-repeat;
  height: 100vh;
  margin: 0;  
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

h1 {
  font-size: 3em;
  color: #8f0330;
  margin-top: -30px;
}

h2 {
  font-size: 1.8em;
  color: #007BFF;
  margin-bottom: 10px;
}

button {
  background-color: #007BFF;
  color: white;
  font-size: 1em;
  padding: 10px 20px;
  border: none;
  cursor: pointer;
  border-radius: 5px;
  transition: background-color 0.3s ease;
}

button:hover {
  background-color: #0056b3;
}

button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}



.content {
  background-color: rgba(255, 255, 255, 0.8); /* White background with 70% transparency */
  padding: 80px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
  width: 70%; /* Adjust width as needed */
  height: auto; /* Use auto height */
  max-width: 2000px; /* Ensure it's not too wide */
  max-height: 60vh; /* Maximum height is 80% of the viewport height */
  min-height: 280px; /* Minimum height for smaller screens */
  transition: height 0.3s ease-in-out; /* Smooth height transition */
  
}
</style>
