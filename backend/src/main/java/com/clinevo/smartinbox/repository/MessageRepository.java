package com.clinevo.smartinbox.repository;

import com.clinevo.smartinbox.entity.Message;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Collection;
import java.util.Optional;

@Repository
public interface MessageRepository extends JpaRepository<Message, Long> {
    Optional<Message> findWithAttachmentsById(Long id);
    List<Message> findByStatusOrderByReceivedDateDesc(Message.ProcessStatus status);
    List<Message> findByStatusInOrderByReceivedDateDesc(Collection<Message.ProcessStatus> statuses);
    List<Message> findAllByOrderByReceivedDateDesc();
    List<Message> findByStatusIn(Collection<Message.ProcessStatus> statuses);
    boolean existsByExternalMessageId(String externalMessageId);
    List<Message> findBySyntheticDemoTrue();
    boolean existsBySenderEndingWith(String suffix);
    long deleteBySenderEndingWith(String suffix);
        List<Message> findBySender(String sender);
}
