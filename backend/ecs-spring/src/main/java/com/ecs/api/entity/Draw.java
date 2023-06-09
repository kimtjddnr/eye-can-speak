package com.ecs.api.entity;

import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import javax.persistence.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;


@Entity
@Table(name = "Draw")
@Getter
@Setter
@NoArgsConstructor
@EntityListeners(AuditingEntityListener.class)
public class Draw {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name="DRAW_NO")
    private int drawNo;

    @Column(name="DRAW_DRAWING", nullable = false)
    private String drawDrawing;

    @CreatedDate
    @Column(name="DRAW_CREATE_DATE")
    private LocalDateTime drawCreateDate;

    @LastModifiedDate
    @Column(name="DRAW_RECENT_DATE")
    private LocalDateTime drawRecentDate;

    @Column(name="DRAW_POST_TF", nullable = false)
    boolean drawPostTF;

    @ManyToOne
    @JoinColumn(name = "USERS_NO")
    private Users usersNo;

    @ManyToOne
    @JoinColumn(name = "CATEGORY_NO")
    private Category categoryNo;

    @OneToMany(mappedBy = "drawNo",cascade = CascadeType.REMOVE)
    private List<Likes> likes = new ArrayList<>();

    @Builder
    public Draw(int drawNo){
        this.drawNo=drawNo;
    }


}
