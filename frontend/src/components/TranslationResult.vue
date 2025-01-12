<template>
  <div class="translation-result">
    <h2>Your Translation:</h2>
    <p>{{ result }}</p>
  </div>
</template>
  
  <script>
  export default {
    name: 'TranslationResult',
    props: {
      result: {
        type: String,
        required: true,
      }
    }
  };
  </script>
  
  <style scoped>
  .translation-result {
    max-width: 600px;
    margin: 20px auto;
    padding: 20px;
    /* background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); */
  }

  @media (max-width: 600px) {
    .translation-result {
        padding: 15px;
    }

    .translation-result h2 {
        font-size: 1.5em;
    }

    .translation-result p {
        font-size: 1em;
    }
  }
  
  .translation-result h2 {
    font-size: 1.8em;
    color: black;
  }
  
  .translation-result p {
    font-size: 1.2em;
    line-height: 1.5;
    color: #333;
    word-wrap: break-word;
  }
  </style>
  