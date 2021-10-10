<template>
  <header>
    <img class="logo" src="../assets/img/common/logo.svg" alt="Logo">
    <img class="logo-text" src="../assets/img/common/logo_text.svg" alt="Logo Text">
  </header>
</template>

<script>
export default {

}
</script>

<style scoped>
header {
    height: calc(30vh - 2 * var(--gap));
    margin: var(--gap);
    display: grid;
    grid-template-columns: repeat(10, 1fr);
    grid-template-rows: repeat(3, 1fr);
    gap: var(--gap);
}

.logo {
    height: 30px;
    width: 30px;
    display: block;
}

.logo-text {
    padding: 10px 0;
    border-top: 2px solid black;
    border-bottom: 2px solid black;
    grid-column: 1 / span 5;
    grid-row: 2 / span 2;
    width: 100%;
}

</style>
