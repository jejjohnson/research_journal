window.MathJax = {
    options: {
        ignoreHtmlClass: 'tex2jax_ignore',
        processHtmlClass: 'tex2jax_process',
        renderActions: {
            find: [10, function (doc) {
                for (const node of document.querySelectorAll('script[type^="math/tex"]')) {
                    const display = !!node.type.match(/; *mode=display/);
                    const math = new doc.options.MathItem(node.textContent, doc.inputJax[0], display);
                    const text = document.createTextNode('');
                    const sibling = node.previousElementSibling;
                    node.parentNode.replaceChild(text, node);
                    math.start = { node: text, delim: '', n: 0 };
                    math.end = { node: text, delim: '', n: 0 };
                    doc.math.push(math);
                    if (sibling && sibling.matches('.MathJax_Preview')) {
                        sibling.parentNode.removeChild(sibling);
                    }
                }
            }, '']
        }
    }
};
// window.MathJax = {
//     tex2jax: {
//         inlineMath: [["$", "$"], ["\\(", "\\)"]],
//         displayMath: [["$$", "$$"], ["\\[", "\\]"]]
//     },
//     TeX: {
//         TagSide: "right",
//         TagIndent: ".8em",
//         MultLineWidth: "85%",
//         equationNumbers: {
//             autoNumber: "AMS",
//         },
//         unicode: {
//             fonts: "STIXGeneral,'Arial Unicode MS'"
//         }
//     },
//     displayAlign: "center",
//     showProcessingMessages: false,
//     messageStyle: "none"
// };