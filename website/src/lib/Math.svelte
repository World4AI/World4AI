<script>
    import katex from 'katex';
    let math = (latex) => {
        let html = katex.renderToString(latex, {
            throwOnError: false
        });
        return html;
    }

    export let latex;
</script>

{@html math(latex)}
