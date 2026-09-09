package com.clinevo.smartinbox.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@RestController
@RequestMapping("/api/literature")
@RequiredArgsConstructor
public class LiteratureController {

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.service.url:http://localhost:8000/analyze}")
    private String aiServiceUrl;

    @PostMapping("/screen")
    public ResponseEntity<Map<String, Object>> screen(@RequestBody List<Map<String, Object>> cases) {
        String literatureUrl = aiServiceUrl.replace("/analyze", "/literature-screen");

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<List<Map<String, Object>>> request = new HttpEntity<>(cases == null ? List.of() : cases, headers);

        ResponseEntity<Map> response = restTemplate.exchange(
                literatureUrl,
                HttpMethod.POST,
                request,
                Map.class
        );

        Map<String, Object> body = response.getBody() == null ? new LinkedHashMap<>() : response.getBody();
        body.putIfAbsent("status", "ok");
        return ResponseEntity.ok(body);
    }
}
