<template>
  <header>
    <img class="logo" src="../assets/img/common/logo.svg" alt="Logo">
  </header>
</template>

<script>
export default {

}
</script>

<style scoped>
header {
    height: calc(10vh - 2 * var(--gap));
    margin: var(--gap);
    display: flex;
    justify-content: flex-start;
    align-items: center;
}

.logo {
    height: 30px;
    width: 30px;
    display: block;
}

</style>
