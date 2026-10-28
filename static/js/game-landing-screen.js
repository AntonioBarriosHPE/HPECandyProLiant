$(function () {
    let inactivityTimer;
    let splashEnabled = true;
    try {
        splashEnabled = localStorage.getItem("candy.showSplashScreen") !== "false";
    } catch (error) {
        console.warn("Splash screen preference could not be loaded.", error);
    }

    function updateSplashControls() {
        $("#showSplashScreenInput").prop("checked", splashEnabled);
        $("#btnBackToHome").toggle(splashEnabled);
    }

    window.setCandySplashEnabled = function(enabled) {
        if (splashEnabled === enabled) return;
        splashEnabled = enabled;
        try {
            localStorage.setItem("candy.showSplashScreen", String(enabled));
        } catch (error) {
            console.warn("Splash screen preference could not be saved.", error);
        }
        updateSplashControls();
        if (splashEnabled && !window.isDemoRunning) {
            returnToLandingScreen();
        } else {
            $("#splashScreen").stop(true, true).hide();
        }
        resetInactivityTimer();
    };

    updateSplashControls();
    $("#splashScreen").toggle(splashEnabled);
    $("#settingsModal").on("show.bs.modal", updateSplashControls);

    function returnToLandingScreen() {
        if (!splashEnabled) return;
        console.log("Returning to landing screen");

        window.stopCandyGame();

        $("#splashScreen").stop(true, true).fadeIn(500);
    }

    function resetInactivityTimer() {
        clearTimeout(inactivityTimer);
        if (!splashEnabled) return;
        inactivityTimer = setTimeout(function () {
            if (window.isDemoRunning) {
                resetInactivityTimer();
                return;
            }
            if ($("#splashScreen").is(":hidden")) {
                returnToLandingScreen();
            }
        }, 60000);
    }

    $("#playGameBtn").on("click", function () {
        console.log("Play Game clicked; starting the demo");
        $("#splashScreen").stop(true, true).fadeOut(500);
        if (window.playCandyGame) {
            window.playCandyGame();
        }
        resetInactivityTimer();
    });

    $("#btnBackToHome").on("click", returnToLandingScreen);

    // Reset timer whenever the user interacts.
    $(document).on("mousemove click keypress touchstart touchmove", resetInactivityTimer);
});
