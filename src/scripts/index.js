import $ from "jquery";
import { Weights } from './Weights';

$(function() {
    $("h1").on("click", function() {
        alert("jQuery is working!");
    });

    $('#model_size_select').on("change", function(){
        alert($(this).text());
    });

    $('#input_file_weights').on('change', async function() {
        let file = $(this)[0].files[0];
        console.log(file);

        let data = await file.arrayBuffer();
        console.log(data);

        let weights = new Weights(file.name);

        await weights.init(data);

        console.log(weights);

        weights.get('decoder.positional_embedding').print();
    });

    $('#input_file_config').on('change', async function() {
        let file = $(this)[0].files[0];
        console.log(file);

        var fileread = new FileReader();
        fileread.onload = function(e) {
        var content = e.target.result;

        var intern = JSON.parse(content);
        console.log(intern);
        console.log(intern.n_mels);
        };

        fileread.readAsText(file);
    });

});