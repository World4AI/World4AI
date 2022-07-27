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
  <div class="mx-3">
    <div class="flex-container">
      <div class="clip-logo">
        <a href="/"
          ><img class="logo" src="/logo/logo.svg" alt="World4AI Logo" /></a
        >
      </div>
      <div
        class="hamburger"
        class:active={isHamburgerClicked}
        on:click={hamburgerEventHandler}
      >
        <div class="line top" />
        <div class="line mid" />
        <div class="line bot" />
      </div>

      <div
        class="links"
        class:show-nav={isHamburgerClicked}
        class:animate-nav={isHamburgerClicked}
      >
        <nav>
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
                      class:selecteddropdown={path === "/blocks/introduction"}
                      href="/blocks/introduction">Introduction</a
                    >
                  </li>
                  <li>
                    <a
                      class:selecteddropdown={path === "/blocks/deep_learning"}
                      href="/blocks/deep_learning/introduction">Deep Learning</a
                    >
                  </li>
                  <li>
                    <a
                      class:selecteddropdown={path ===
                        "/blocks/reinforcement_learning"}
                      href="/blocks/reinforcement_learning/introduction"
                      >Reinforcement Learning</a
                    >
                  </li>
                </ul>
              </div>
            </li>
            <li>
              <a class:selected={path === "/sponsor"} href="/sponsor">Sponsor</a
              >
            </li>
            <li>
              <a class:selected={path === "/about"} href="/about">About</a>
            </li>
          </ul>
        </nav>
        <div class="social-links">
          <!-- Feather icons -->
          <a href="https://github.com/World4AI" target="_blank">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--background-color)"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="feather feather-github"
              ><path
                d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"
              /></svg
            >
          </a>
          <a href="https://twitter.com" target="_blank">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--background-color)"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="feather feather-twitter"
              ><path
                d="M23 3a10.9 10.9 0 0 1-3.14 1.53 4.48 4.48 0 0 0-7.86 3v1A10.66 10.66 0 0 1 3 4s-4 9 5 13a11.64 11.64 0 0 1-7 2c9 5 20 0 20-11.5a4.5 4.5 0 0 0-.08-.83A7.72 7.72 0 0 0 23 3z"
              /></svg
            >
          </a>
          <a
            href="https://www.youtube.com/channel/UCARDap4Qtr6Jsus1Ke53uaQ/featured"
            target="_blank"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--background-color)"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="feather feather-youtube"
              ><path
                d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z"
              /><polygon
                points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02"
              /></svg
            >
          </a>
        </div>
      </div>
    </div>
  </div>
</header>

<style>
  .clip-logo {
    clip-path: polygon(35% 0, 100% 0, 65% 100%, 0 100%);
    background: var(--background-color);
    height: 100%;
    width: 80px;
    flex-shrink: 0;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  header {
    background-color: var(--text-color);
    position: fixed;
    left: 0;
    top: 0;
    right: 0;
    z-index: 10;
  }

  .flex-container {
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .links {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
  }

  .social-links {
    margin-left: 80px;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .social-links a {
    margin-left: 10px;
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
    color: var(--background-color);
    text-decoration: none;
    text-transform: uppercase;
  }

  .nav {
    position: relative;
  }

  .nav::before,
  .nav::after {
    content: "";
    position: absolute;
    width: 15px;
    height: 1px;
    background: var(--background-color);
    top: 50%;
    z-index: 1;
  }

  .nav::before {
    left: -10px;
  }

  .nav::after {
    right: 0;
  }

  .selected {
    position: relative;
  }

  .selected::before {
    position: absolute;
    content: "";
    height: 2px;
    width: 100%;
    top: 25px;
    left: 0%;
    background-color: var(--background-color);
  }

  .selecteddropdown {
    position: relative;
  }

  .selecteddropdown::before {
    position: absolute;
    content: "";
    height: 5px;
    width: 5px;
    top: 50%;
    left: -15px;
    background-color: var(--background-color);
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
    background-color: var(--text-color);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .dropdown-content li {
    margin-top: 20px;
    margin-bottom: 20px;
    margin-left: 30px;
    width: 100%;
  }

  .dropdown:hover .dropdown-content {
    display: block;
  }

  .hamburger {
    display: none;
  }

  @media (max-width: 1000px) {
    .links {
      padding-top: 50px;
      z-index: 999;
      position: fixed;
      right: 100%;
      top: 100px;
      background: var(--text-color);
      flex-direction: column;
      transition: all 0.5s;
    }

    .animate-nav {
      transform: translateX(100%);
    }

    .nav {
      flex-direction: column;
      align-items: flex-start;
      justify-content: flex-start;
      width: 100vw;
      height: 95vh;
      padding-left: 10px;
    }

    .nav::before,
    .nav::after {
      display: none;
    }

    nav > .nav > li {
      margin-bottom: 20px;
    }

    .dropdown-content {
      display: block;
      position: relative;
    }
    .dropdown-content li {
      margin-top: 20px;
      margin-bottom: 10px;
      margin-left: 0px;
    }

    .dropdown-content ul {
      border: none;
      background-color: var(--aside-color);
      margin-left: 15px;
    }

    .social-links {
      display: none;
    }

    /* hamburger section */
    .hamburger {
      height: 15px;
      width: 25px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: center;
    }

    .hamburger.active {
      height: 25px;
    }

    .hamburger:hover {
      cursor: pointer;
    }

    .hamburger:not(.active) .top {
      transform: translate(5px);
    }

    .hamburger:not(.active) .bot {
      transform: translate(5px);
    }

    .line {
      height: 1px;
      width: 100%;
      background-color: var(--background-color);
      transition: all 0.5s ease-in-out;
      pointer-events: none;
    }

    .active {
      transform: translateX(-10px);
    }

    .active .top {
      transform: rotate(45deg) translate(8px, 9px) scale(1.3);
    }

    .active .bot {
      transform: rotate(-45deg) translate(8px, -9px) scale(1.3);
    }

    .active .mid {
      transform: translate(-27px, 0);
      opacity: 0;
    }
  }
</style>
