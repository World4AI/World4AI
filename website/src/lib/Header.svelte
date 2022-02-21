<script>
  import { page } from "$app/stores";

  $: path = (() => {
    let pathname = $page.url.pathname;
    let splitPath = pathname.split("/");
    if (splitPath.length == 1) {
      return "/";
    } else if (splitPath.length == 2) {
      return "/" + pathname.split("/")[1];
    } else {
      return "/" + pathname.split("/")[1] + "/" + pathname.split("/")[2];
    }
  })();

  let isHamburgerClicked = false;

  function hamburgerEventHandler() {
    isHamburgerClicked = !isHamburgerClicked;
  }
</script>

<header>
  <a href="/"><img class="logo" src="/logo/logo.svg" alt="World4AI Logo" /></a>
  <div
    class="hamburger"
    class:active={isHamburgerClicked}
    on:click={hamburgerEventHandler}
  >
    <div class="line top" />
    <div class="line mid" />
    <div class="line bot" />
  </div>
  <nav class:show-nav={isHamburgerClicked}>
    <ul class="nav">
      <li><a class:selected={path === "/"} href="/">Home</a></li>
      <li class="dropdown">
        <a
          class:selected={path.split("/")[1] === "blocks"}
          href="/blocks/introduction">Blocks</a
        >
        <div class="dropdown-content">
          <ul>
            <li>
              <a
                class:selected={path === "/blocks/introduction"}
                href="/blocks/introduction">Introduction</a
              >
            </li>
            <li>
              <a
                class:selected={path === "/blocks/programming"}
                href="/blocks/programming/introduction">Programming</a
              >
            </li>
            <li>
              <a
                class:selected={path === "/blocks/mathematics"}
                href="/blocks/mathematics/introduction">Mathematics</a
              >
            </li>
            <li>
              <a
                class:selected={path === "/blocks/deep_learning"}
                href="/blocks/deep_learning">Deep Learning</a
              >
            </li>
            <li>
              <a
                class:selected={path === "/blocks/reinforcement_learning"}
                href="/blocks/reinforcement_learning/introduction"
                >Reinforcement Learning</a
              >
            </li>
          </ul>
        </div>
      </li>
      <li><a class:selected={path === "/about"} href="/about">About</a></li>
      <li>
        <a class:selected={path === "/sponsor"} href="/sponsor">Sponsor</a>
      </li>
    </ul>
    <ul class="external-links">
      <li>
        <a href="https://github.com/World4AI/World4AI" target="_blank">
          <svg
            width="24px"
            height="24px"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
            ><g data-name="Layer 2"
              ><rect
                width="24"
                height="24"
                transform="rotate(180 12 12)"
                opacity="0"
              /><path
                d="M12 1A10.89 10.89 0 0 0 1 11.77 10.79 10.79 0 0 0 8.52 22c.55.1.75-.23.75-.52v-1.83c-3.06.65-3.71-1.44-3.71-1.44a2.86 2.86 0 0 0-1.22-1.58c-1-.66.08-.65.08-.65a2.31 2.31 0 0 1 1.68 1.11 2.37 2.37 0 0 0 3.2.89 2.33 2.33 0 0 1 .7-1.44c-2.44-.27-5-1.19-5-5.32a4.15 4.15 0 0 1 1.11-2.91 3.78 3.78 0 0 1 .11-2.84s.93-.29 3 1.1a10.68 10.68 0 0 1 5.5 0c2.1-1.39 3-1.1 3-1.1a3.78 3.78 0 0 1 .11 2.84A4.15 4.15 0 0 1 19 11.2c0 4.14-2.58 5.05-5 5.32a2.5 2.5 0 0 1 .75 2v2.95c0 .35.2.63.75.52A10.8 10.8 0 0 0 23 11.77 10.89 10.89 0 0 0 12 1"
                data-name="github"
              /></g
            >
          </svg>
        </a>
      </li>
    </ul>
  </nav>
</header>

<style>
  .hamburger {
    display: none;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .dropdown {
    position: relative;
  }

  .dropdown-content {
    position: absolute;
    display: none;
    z-index: 1;
  }

  .dropdown-content ul {
    margin-top: 20px;
    background-color: var(--background-color);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .dropdown-content li {
    margin-top: 20px;
    margin-bottom: 10px;
    margin-left: 30px;
  }

  .dropdown:hover .dropdown-content {
    display: block;
  }

  .logo {
    height: 30px;
    width: 30px;
  }

  nav {
    display: flex;
    justify-content: space-between;
  }

  nav > ul {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  li {
    list-style: none;
    margin: 0 20px;
    letter-spacing: 5px;
  }

  a {
    color: var(--text-color);
    text-decoration: none;
    text-transform: uppercase;
  }

  nav a:hover {
    color: white;
  }

  .selected {
    position: relative;
  }

  .selected::before {
    position: absolute;
    content: "";
    height: 8px;
    width: 8px;
    top: 5px;
    left: -20px;
    background-color: var(--text-color);
  }

  svg {
    fill: var(--text-color);
  }

  @media (max-width: 768px) {
    nav {
      z-index: 999;
      display: none;
      position: absolute;
      right: 0;
      top: 60px;
      background: var(--aside-color);
      flex-direction: column;
    }

    .nav {
      flex-direction: column;
      align-items: flex-start;
      justify-content: space-around;
      width: 100vw;
      height: 95vh;
      padding-left: 10px;
    }

    .external-links {
      display: none;
    }

    .dropdown-content {
      display: block;
      position: relative;
    }

    .dropdown-content ul {
      border: none;
      background-color: var(--aside-color);
    }

    .show-nav {
      display: block;
    }

    .hamburger {
      height: 25px;
      width: 25px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: center;
    }

    .line {
      height: 1px;
      width: 100%;
      background-color: var(--text-color);
      transition: all 1s ease-in-out;
    }

    .active {
      transform: translateX(-10px);
    }

    .active .top {
      transform: rotate(45deg) translate(17px, 0) scale(1.3);
    }

    .active .bot {
      transform: rotate(-45deg) translate(17px, 0) scale(1.3);
    }

    .active .mid {
      opacity: 0;
    }
  }
</style>
