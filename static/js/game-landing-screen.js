$(function () {
    let inactivityTimer;

    function returnToLandingScreen() {
        console.log("Returning to landing screen");

        window.stopCandyGame();

        $("#splashScreen").fadeIn(500);
    }

    function resetInactivityTimer() {
        clearTimeout(inactivityTimer);
        inactivityTimer = setTimeout(function () {
            if ($("#splashScreen").is(":hidden")) {
                returnToLandingScreen();
            }
        }, 60000);
    }

    $("#playGameBtn").on("click", function () {
        console.log("Play Game clicked; starting the demo");
        $("#splashScreen").fadeOut(500);
        if (window.playCandyGame) {
            window.playCandyGame();
        }
        resetInactivityTimer();
    });

    $("#btnBackToHome").on("click", returnToLandingScreen);

    // Reset timer whenever the user interacts.
    $(document).on("mousemove click keypress touchstart touchmove", resetInactivityTimer);
});
