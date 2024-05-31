document.addEventListener("DOMContentLoaded", function() {
    // This code will run after the document is fully loaded
    console.log("Document fully loaded and parsed");

    // Example: Show an alert when the image file is selected
    const fileInput = document.getElementById('imageUpload');
    fileInput.addEventListener('change', function(event) {
        if (fileInput.files.length > 0) {
            const fileName = fileInput.files[0].name;
            alert('File ' + fileName + ' has been selected.');
        }
    });
});
