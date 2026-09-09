FROM gradle:8.10-jdk21 AS build

WORKDIR /workspace

COPY build.gradle settings.gradle ./

RUN gradle --no-daemon dependencies

COPY src src

RUN gradle --no-daemon bootJar

FROM eclipse-temurin:21-jre

WORKDIR /app

COPY --from=build /workspace/build/libs/*.jar app.jar

ENV SPRING_PROFILES_ACTIVE=container

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "/app/app.jar"]
