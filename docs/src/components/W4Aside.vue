<template>
  <aside>
    <div class="aside-top">
      <div class="hamburger-wrapper">
          <div class="hamburger">
              <span></span>
              <span></span>
              <span></span>
          </div>
      </div>
    </div>
    <div class="aside-bottom">
      <W4SideNav class="nav"/>
    </div>
  </aside>
</template>

<script>
import W4SideNav from './sidebar/W4SideNav.vue'

export default {
  components: {
    W4SideNav
  }
}
</script>

<style scoped>
.aside-top {
    height: calc(30vh - 2 * var(--gap));
    margin: var(--gap);
    display: flex;
    justify-content: flex-end;
}

.hamburger {
    cursor: pointer;
}

.hamburger span {
    display: block;
    height: 1px;
    width: 20px;
    background-color: white;
    margin-bottom: 0.5rem;
}

.aside-bottom {
    height: calc(70vh - var(--gap));
    margin: 0 var(--gap) var(--gap) var(--gap);
}
nav {
  padding-left: 0.5rem;
}

</style>