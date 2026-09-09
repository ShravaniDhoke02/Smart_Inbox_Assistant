import { of } from 'rxjs';
import { AppComponent } from './app.component';

describe('AppComponent', () => {
  it('translates the selected message text', () => {
    const service = {
      translate: jasmine.createSpy('translate').and.returnValue(of({ translatedText: 'Translated test text' }))
    } as any;
    const sanitizer = {} as any;
    const component = new AppComponent(service, sanitizer);
    component.selected = { body: 'Texto de prueba', language: 'Spanish' };

    component.translateSelected();

    expect(service.translate).toHaveBeenCalledWith('Texto de prueba');
    expect(component.translation).toBe('Translated test text');
    expect(component.translating).toBeFalse();
  });

});