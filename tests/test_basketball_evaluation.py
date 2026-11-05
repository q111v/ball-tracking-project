from tools.evaluate_basketball import evaluate


def row(frame,center,source='test'):
    x,y=center
    return dict(frame=str(frame),id='1',x1=str(x-5),y1=str(y-5),x2=str(x+5),y2=str(y+5),source=source)


def test_evaluation_does_not_hide_missing_frames_or_wrong_heads():
    checks=dict(tolerance_px=12,frames=[dict(frame=0,center=[100,100]),dict(frame=1,center=[110,100])],
                negative_checks=[dict(frame=2,object='head',bbox=[190,190,210,210])])
    out=evaluate([row(0,[100,100]),row(1,[0,0],'missing'),row(2,[200,200])],checks)
    assert out['visible_center_checks']==2
    assert out['visible_center_hits']==1
    assert out['wrong_object_hits']==1
    assert out['mean_center_error_px']==0
