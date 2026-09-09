package com.clinevo.smartinbox.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.servlet.view.RedirectView;

@Controller
public class FrontendRedirectController {
    @Value("${frontend.url:http://localhost:4200}")
    private String frontendUrl;

    @GetMapping({"/", "/index.html"})
    public RedirectView redirectToFrontend() {
        return new RedirectView(frontendUrl, true);
    }
}
