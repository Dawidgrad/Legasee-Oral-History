function build_div_object(id, class_name, parent_id){
    var div = document.createElement("div");
    div.id = id;
    div.className = class_name;
    document.getElementById(parent_id).appendChild(div);
    return div;
}

function add_title_div(id, class_name, parent_id, title){
    // center position
    let div = build_div_object(id, class_name, parent_id);
    div.style.textAlign = "center";
    div.style.fontSize = "30px";
    div.style.fontWeight = "bold";
    document.getElementById(id).innerHTML = title;
}

function build_str(list_of_json){
    let str = "";
    for (let i = 0; i < list_of_json.length; i++){
        //quaternary
        str_txt = list_of_json[i].entities != null ? `<b>${list_of_json[i].text}</b>` : list_of_json[i].text+" ";
        str_txt = list_of_json[i].speaker[0] == 1 ? `<span class="interviewee">${str_txt}</span>` : `<span class="interviewer">${str_txt}</span>`;
        str += str_txt+" ";
    }
    return str;
}

function main(){
    // json = data; loaded in index.html
    transcript_data = data
    console.log(transcript_data);
    add_title_div("page_title", "titletxt", "topdiv", `${data.file_name.replace('_', ' ')}`);
    contents = build_str(transcript_data.contents);
    document.getElementById("transcript").innerHTML = contents;
}


document.addEventListener("DOMContentLoaded", function(event) {
    main();
});