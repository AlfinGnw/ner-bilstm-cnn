document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("detectForm");
    const btn = document.getElementById("btnDeteksi");
    const spinner = document.getElementById("loading");
    const lokasiTambahanInput = document.getElementById("lokasi_tambahan");
    const teks = document.querySelector('textarea[name="text"]').value;

    if (!form || !btn || !lokasiTambahanInput || !teks.trim()) return;

    btn.addEventListener("click", function (e) {
        e.preventDefault();

        spinner.classList.remove("d-none");
        btn.setAttribute("disabled", "disabled");

        puter.ai.chat(
            `次のテキストから場所を抽出してください。抽出は町や村レベルまでに制限してください。県、市、国は表示しないでください。1行に1つの場所を記載してください。\n\n${teks}`,
            {
                model: "gpt-4.1-nano",
                temperature: 0.3
            }
        ).then(res => {
            let hasil = res.message?.content;
            if (Array.isArray(hasil)) hasil = hasil[0]?.text || '';
            const lokasi = hasil.trim().split("\n").map(x => x.trim()).filter(x => x);
            lokasiTambahanInput.value = lokasi.join("|");
            form.submit();
        }).catch(err => {
            console.error("Ekstraksi lokasi gagal:", err);
            spinner.classList.add("d-none");
            btn.removeAttribute("disabled");
        });
    });
});
