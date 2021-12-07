<script>
    import katex from 'katex';
    export let latex;
    const math = (formula) => {
        let html = katex.renderToString(formula, {
            throwOnError: false
        });
        return html;
    }
    
    $: result = math(latex)
</script>

<span>{@html result}</span>
