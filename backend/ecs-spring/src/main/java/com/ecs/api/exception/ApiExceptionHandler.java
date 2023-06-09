package com.ecs.api.exception;

import com.ecs.api.exception.User.LoginFailException;
import com.ecs.api.exception.User.PasswordNotMatchException;
import com.ecs.api.exception.User.UserAlreadyExistException;
import com.ecs.api.exception.User.UserNotExistException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

import java.time.ZoneId;
import java.time.ZonedDateTime;

@ControllerAdvice
public class ApiExceptionHandler {

    @ExceptionHandler(value = {EmptyValueExistException.class})
    public ResponseEntity<Object> handleEmptyValueException(EmptyValueExistException e){

        HttpStatus httpStatus = HttpStatus.BAD_REQUEST;

        ApiException apiException = new ApiException(
                ExceptionMessage.EMPTY_VALUE_EXCEPTION_MESSAGE,
                httpStatus,
                ZonedDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return new ResponseEntity<>(apiException, httpStatus);
    }
    // 예시 -----------------------------------------------------------------------------------------
    @ExceptionHandler(value = {PasswordNotMatchException.class})
    public ResponseEntity<Object> handlePasswordNotMatchException(PasswordNotMatchException e){

        HttpStatus httpStatus = HttpStatus.BAD_REQUEST;

        ApiException apiException = new ApiException(
                ExceptionMessage.PASSWORD_NOT_MATCH_MESSAGE,
                httpStatus,
                ZonedDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return new ResponseEntity<>(apiException, httpStatus);
    }

    @ExceptionHandler(value = {UserAlreadyExistException.class})
    public ResponseEntity<Object> handleUserAlreadyExistException(UserAlreadyExistException e){

        HttpStatus httpStatus = HttpStatus.BAD_REQUEST;

        ApiException apiException = new ApiException(
                ExceptionMessage.USER_ALREADY_EXIST_MESSAGE,
                httpStatus,
                ZonedDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return new ResponseEntity<>(apiException, httpStatus);
    }

    @ExceptionHandler(value = {UserNotExistException.class})
    public ResponseEntity<Object> handleUserNotExistException(UserNotExistException e){

        HttpStatus httpStatus = HttpStatus.BAD_REQUEST;

        ApiException apiException = new ApiException(
                ExceptionMessage.USER_NOT_EXIST_MESSAGE,
                httpStatus,
                ZonedDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return new ResponseEntity<>(apiException, httpStatus);
    }

    @ExceptionHandler(value = {LoginFailException.class})
    public ResponseEntity<Object> handleLoginFailException(LoginFailException e){

        HttpStatus httpStatus = HttpStatus.BAD_REQUEST;

        ApiException apiException = new ApiException(
                ExceptionMessage.LOG_IN_FAIL_MESSAGE,
                httpStatus,
                ZonedDateTime.now(ZoneId.of("Asia/Seoul"))
        );

        return new ResponseEntity<>(apiException, httpStatus);
    }
}