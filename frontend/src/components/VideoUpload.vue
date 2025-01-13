<template>
  <div class="video-upload">
    <h2>Upload Your Sign Language Video for Translation</h2>
    <input 
      type="file"
      accept="video/*" 
      @change="onFileChange"
    >
    <button 
      :disabled="!videoFile" 
      @click="uploadVideo" 
    >
      Upload Video
    </button>
  </div>
</template>
  
  <script>
  export default {
    name: 'VideoUpload',
    emits: ['video-uploaded'],
    data() {
      return {
        videoFile: null,
      };
    },
    methods: {
      onFileChange(event) {
        this.videoFile = event.target.files[0];
      },
      async uploadVideo() {
        if (!this.videoFile) return;
  
        const formData = new FormData();
        formData.append('video', this.videoFile);
  
        try {
          // Replace with your backend API endpoint
        //   const response = await fetch('http://localhost:5000/api/translate', {
            const response = await fetch('http://localhost:3000/upload', {
            method: 'POST',
            body: formData,
          });
  
          const data = await response.json();
          this.$emit('video-uploaded', data.translation);  // Pass the result to App.vue
        } catch (error) {
          console.error('Error uploading video:', error);
        }
         }
    }
  };
  </script>
  
  <style scoped>
  .video-upload {
    max-width: 600px;
    margin: 0 auto;
    padding: 20px;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }

  @media (max-width: 600px) {
    .video-upload input {
        padding: 8px;
        font-size: 1em;
    }

    .video-upload button {
        font-size: 1em;
        padding: 10px;
    }
  }
  
  .video-upload input {
    display: block;
    margin: 20px 0;
    padding: 10px;
    width: 96.5%;
    border: 1px solid #ccc;
    border-radius: 5px;
  }
  
  .video-upload button {
    width: 100%;
    padding: 12px;
    font-size: 1.1em;
  }
  </style>
  