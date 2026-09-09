package com.clinevo.smartinbox.service;
import com.clinevo.smartinbox.entity.AuditLog;
import com.clinevo.smartinbox.entity.Message;
import com.clinevo.smartinbox.repository.MessageRepository;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class AuditService {
	private final MessageRepository messages;
	private final ObjectMapper objectMapper;

	public void log(String type, Long id, String action, String actor, String details) {
		messages.findById(id).ifPresent(message -> {
			try {
				List<AuditLog> logs = message.getAuditLogsJson() == null
						? new ArrayList<>()
						: objectMapper.readValue(message.getAuditLogsJson(), new TypeReference<>() {});
				AuditLog entry = new AuditLog();
				entry.setId((long) logs.size() + 1);
				entry.setEntityType(type);
				entry.setEntityId(id);
				entry.setAction(action);
				entry.setPerformedBy(actor);
				entry.setTimestamp(LocalDateTime.now());
				entry.setDetails(details);
				logs.add(0, entry);
				message.setAuditLogsJson(objectMapper.writeValueAsString(logs));
				messages.save(message);
			} catch (Exception error) {
				throw new IllegalStateException("Could not persist audit JSON", error);
			}
		});
	}
}
