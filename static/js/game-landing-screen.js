$(function () {

    $("#playGameBtn").on("click", function () {

        console.log("Play Game clicked");

        $("#splashScreen").fadeOut(500);

        // Start the existing application
        $("#btnStartDemo").trigger("click");

    });

    $("#btnBackToHome").on("click", function () {

        console.log("Back to Home clicked");

        // Stop game if running
        if ($("#btnStartDemo").text().includes("Stop")) {
            $("#btnStartDemo").trigger("click");
        }

        // Show landing screen
        $("#splashScreen").fadeIn(500);

    });
});

$(function () {

    let inactivityTimer;

    function returnToLandingScreen() {

        console.log("Returning to landing screen");

        // Stop the game if running
        if ($("#btnStartDemo").text().includes("Stop")) {
            $("#btnStartDemo").trigger("click");
        }

        $("#splashScreen").fadeIn(500);
    }

    function resetInactivityTimer() {

        clearTimeout(inactivityTimer);

        inactivityTimer = setTimeout(function () {

            // Only return if user is currently in the game
            if ($("#splashScreen").is(":hidden")) {
                returnToLandingScreen();
            }

        }, 60000);   // 60 seconds
    }

    $("#playGameBtn").on("click", function () {

        console.log("Play Game clicked");

        $("#splashScreen").fadeOut(500);

        $("#btnStartDemo").trigger("click");

        resetInactivityTimer();
    });

    $("#btnBackToHome").on("click", function () {

        returnToLandingScreen();
    });

    // Reset timer whenever the user interacts
    $(document).on(
        "mousemove click keypress touchstart touchmove",
        function () {
            resetInactivityTimer();
        }
    );

});